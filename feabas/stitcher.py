from collections import defaultdict
from concurrent.futures.process import ProcessPoolExecutor
from concurrent.futures import as_completed
import cv2
from functools import partial
import gc
from multiprocessing import get_context
import numpy as np
import os
from rtree import index
from scipy.interpolate import interp1d
from scipy import sparse
import scipy.sparse.csgraph as csgraph

from feabas.dal import StaticImageLoader
from feabas.matcher import stitching_matcher
from feabas.mesh import Mesh
from feabas.optimizer import SLM
from feabas import miscs
from feabas.constant import *


class Stitcher:
    """
    Class to handle everything related to 2d stitching.

    Args:
        imgpaths(list): fullpaths of the image file list
        bboxes(N x 4 ndarray): initial estimation of the bounding boxes for each
            image tiles. Could be from the microscope stage coordinates or rough
            stitching results. (xmin, ymin, xmax, ymax), supposed integers.
    Kwargs:
        root_dir(str): if provided, imgpaths arg is considered to be relative
            paths and root_dir will be prepended to generate the fullpaths.
    """
    def __init__(self, imgpaths, bboxes, **kwargs):
        root_dir = kwargs.get('root_dir', None)
        groupings = kwargs.get('groupings', None)
        if bool(root_dir):
            self.imgrootdir = root_dir
            self.imgrelpaths = imgpaths
        else:
            self.imgrootdir = os.path.dirname(os.path.commonprefix(imgpaths))
            self.imgrelpaths = [os.path.relpath(s, self.imgrootdir) for s in imgpaths]
        bboxes = np.round(bboxes).astype(np.int32)
        self.init_bboxes = bboxes
        self.tile_sizes = miscs.bbox_sizes(bboxes)
        self.average_tile_size = np.median(self.tile_sizes, axis=0)
        init_offset = bboxes[...,:2]
        self._init_offset = init_offset - init_offset.min(axis=0)
        self.matches = {}
        self.match_strains = {}
        self.meshes = None
        self.mesh_sharing = np.arange(self.num_tiles)
        self._optimizer = None
        self._default_mesh_cache = None
        self.set_groupings(groupings)


    @classmethod
    def from_coordinate_file(cls, filename, **kwargs):
        """
        initialize Stitcher from a coordinate txt file. Each row in the file
        follows the pattern:
            image_path  x_min  y_min  x_max(optional)  y_max(optional)
        if x_max and y_max is not provided, they are inferred from tile_size.
        If relative path is provided in the image_path colume, at the first line
        of the file, the root_dir can be defined as:
            {ROOT_DIR}  rootdir_to_the_path
            {TILE_SIZE} tile_height tile_width
        Args:
            filename(str): full path to the coordinate file.
        Kwargs:
            rootdir: if the imgpaths colume in the file is relative paths, can
                use this to prepend the paths. Set to None to disable.
            tile_size: the tile size used to compute the bounding boxes in the
                absense of x_max and y_max in the file. If None, will read an
                image file to figure out
            delimiter: the delimiter to separate each colume in the file. If set
                to None, any whitespace will be considered.
        """
        root_dir = kwargs.get('root_dir', None)
        tile_size = kwargs.get('tile_size', None)
        delimiter = kwargs.get('delimiter', '\t') # None for any whitespace
        imgpaths = []
        bboxes = []
        with open(filename, 'r') as f:
            lines = f.readlines()
        if len(lines) == 0:
            raise RuntimeError(f'empty file: {filename}')
        start_line = 0
        for line in lines:
            if '{ROOT_DIR}' in line:
                start_line += 1
                tlist = line.strip().split(delimiter)
                if len(tlist) >= 2:
                    root_dir = tlist[1]
            elif '{TILE_SIZE}' in line:
                start_line += 1
                tlist = line.strip().split(delimiter)
                if len(tlist) == 2:
                    tile_size = (int(tlist[1]), int(tlist[1]))
                elif len(tlist) > 2:
                    tile_size = (int(tlist[1]), int(tlist[2]))
                else:
                    continue
            else:
                break
        relpath = bool(root_dir)
        for line in lines[start_line:]:
            line = line.strip()
            tlist = line.split(delimiter)
            if len(tlist) < 3:
                # raise RuntimeError(f'corrupted coordinate file: {filename}')
                continue
            mpath = tlist[0]
            x_min = float(tlist[1])
            y_min = float(tlist[2])
            if (len(tlist) >= 5) and (tile_size is None):
                x_max = float(tlist[3])
                y_max = float(tlist[4])
            else:
                if tile_size is None:
                    if relpath:
                        mpath_f = os.path.join(root_dir, mpath)
                    else:
                        mpath_f = mpath
                    img = cv2.imread(mpath_f,cv2.IMREAD_GRAYSCALE)
                    tile_size = img.shape
                x_max = x_min + tile_size[-1]
                y_max = y_min + tile_size[0]
            imgpaths.append(mpath)
            bboxes.append((x_min, y_min, x_max, y_max))
        return cls(imgpaths, bboxes, root_dir=root_dir)


    def dispatch_matchers(self, **kwargs):
        """
        run matching between overlapping tiles.
        """
        overwrite = kwargs.get('overwrite', False)
        num_workers = kwargs.get('num_workers', 1)
        min_width = kwargs.get('min_width', 0)
        margin = kwargs.get('margin', 1.0)
        image_to_mask_path = kwargs.get('image_to_mask_path', None)
        matcher_config = kwargs.get('matcher_config', {})
        loader_config = kwargs.get('loader_config', {})
        if bool(loader_config.get('cache_size', None)) and (num_workers > 1):
            loader_config['cache_size'] = int(np.ceil(loader_config['cache_size'] / num_workers))
        loader_config['number_of_channels'] = 1 # only gray-scale matching are supported
        target_func = partial(Stitcher.match_list_of_overlaps,
                              root_dir=self.imgrootdir,
                              min_width=min_width,
                              margin=margin,
                              image_to_mask_path=image_to_mask_path,
                              loader_config=loader_config,
                              matcher_config=matcher_config)
        if overwrite:
            self.matches = {}
            self.match_strains = {}
            overlaps = self.overlaps
        else:
            overlaps = self.overlaps_without_matches
        num_overlaps = len(overlaps)
        if ((num_workers is not None) and (num_workers <= 1)) or (num_overlaps <= 1):
            new_matches, match_strains = target_func(overlaps, self.imgrelpaths, self.init_bboxes)
            self.matches.update(new_matches)
            self.match_strains.update(match_strains)
            return len(new_matches)
        num_workers = min(num_workers, num_overlaps)
        # 180 roughly number of overlaps in an MultiSEM mFoV
        num_overlaps_per_job = min(num_overlaps//num_workers, 180)
        N_jobs = round(num_overlaps / num_overlaps_per_job)
        indx_j = np.linspace(0, num_overlaps, num=N_jobs+1, endpoint=True)
        indx_j = np.unique(np.round(indx_j).astype(np.int32))
        # divide works
        jobs = []
        num_new_matches = 0
        with ProcessPoolExecutor(max_workers=num_workers, mp_context=get_context('spawn')) as executor:
            for idx0, idx1 in zip(indx_j[:-1], indx_j[1:]):
                ovlp_g = overlaps[idx0:idx1] # global indices of overlaps
                mapper, ovlp = np.unique(ovlp_g, return_inverse=True, axis=None)
                ovlp = ovlp.reshape(ovlp_g.shape)
                bboxes = self.init_bboxes[mapper]
                imgpaths = [self.imgrelpaths[s] for s in mapper]
                job = executor.submit(target_func, ovlp, imgpaths, bboxes, index_mapper=mapper)
                jobs.append(job)
            for job in as_completed(jobs):
                matches, match_strains = job.result()
                num_new_matches += len(matches)
                self.matches.update(matches)
                self.match_strains.update(match_strains)
        return num_new_matches


    def find_overlaps(self):
        overlaps = []
        tree = index.Index(interleaved=True)
        for k, bbox in enumerate(self.init_bboxes):
            hits = list(tree.intersection(bbox, objects=False))
            tree.insert(k, bbox, obj=None)
            if bool(hits):
                overlaps.extend([(k, hit) for hit in hits])
        overlaps = np.array(overlaps)
        bboxes0 = self.init_bboxes[overlaps[:,0]]
        bboxes1 = self.init_bboxes[overlaps[:,1]]
        bbox_ov, _ = miscs.bbox_intersections(bboxes0, bboxes1)
        ov_cntr = miscs.bbox_centers(bbox_ov)
        average_step_size = self.average_tile_size[::-1] / 2
        ov_indices = np.round((ov_cntr - ov_cntr.min(axis=0))/average_step_size)
        z_order = miscs.z_order(ov_indices)
        return overlaps[z_order]


    def initialize_meshes(self, mesh_sizes, **kwargs):
        """
        initialize meshes of each tile for mesh relaxation.
        Args:
            mesh_sizes(list): a list of options for mesh size to choose from.
                the actual mesh sizes for each tile will be determined based on
                how much the tile needs to be deformed during the matching.
                Provide only one value to fixed mesh size.
        Kwargs:
            border_width(float): the width of the overlap in the ratio to the
                tile size, so that mesh triangles away from the border can be
                coarser to save computation. Set to None to automatically
                determine.
            interior_growth(float): increase of the mesh size in the interior
                region.
            match_soften(tuple): If set, the meshes will be soften based on
                the strain during matching step. the tuple has two elements:
                (upper_strain, lower_soft_factor), so that the soften factor
                linearly changes from 1 to lower_soft_factor with a strain
                from 0 to upper_strain.
            soft_top(float): the multiplier to apply to the stiffness to the top
                part of the tile (there might be more distorsion at the beginning
                of the scan, so make it weight less).
            soft_top_width(float): the width of the soft top region with regard
                to the tile height.
            soft_left, soft_left_width: same as the "soft top" but in horizontal
                direction.
            cache_size: size of cache that save the intermediate results of
                the meshes.
        """
        border_width = kwargs.get('border_width', None)
        interior_growth = kwargs.get('interior_growth', 3.0)
        match_soften = kwargs.get('match_soften', None)
        soft_top = kwargs.get('soft_top', 0.2)
        soft_top_width = kwargs.get('soft_top_width', 0.1)
        soft_left = kwargs.get('soft_left', 0.2)
        soft_left_width = kwargs.get('soft_top_width', 0.0)
        cache_size = kwargs.get('cache_size', None)
        groupings = self.groupings(normalize=True)
        if (not kwargs.get('force_update', False)) and (self.meshes is not None):
            return
        if (not hasattr(mesh_sizes, '__len__')) or (len(mesh_sizes) == 1) or (len(self.matches) == 0):
            tile_mesh_sizes = np.full(self.num_tiles, np.max(mesh_sizes))
        else:
            # determine the mesh size based on the deformation during mathcing.
            err_contant = 5
            mesh_sizes = np.array(mesh_sizes, copy=False)
            mx_meshsz = np.max(mesh_sizes)
            lower_strain = 0.5 * err_contant / mx_meshsz
            strain_list = list(self.match_strains.items())
            overlap_indices = np.array([s[0] for s in strain_list])
            strain_vals = np.array([(s[1], s[1]) for s in strain_list])
            tile_strains = np.zeros(self.num_tiles, dtype=np.float32)
            np.maximum.at(tile_strains, overlap_indices, strain_vals)
            targt_mesh_size = err_contant / (tile_strains.clip(lower_strain, None))
            mesh_size_diff = np.abs(mesh_sizes.reshape(-1,1) - targt_mesh_size)
            mesh_size_idx = np.argmin(mesh_size_diff, axis=0)
            tile_mesh_sizes = mesh_sizes[mesh_size_idx]
            grp_u, cnt = np.unique(groupings, return_counts=True)
            grp_u = grp_u[cnt>1]
            for g in grp_u:
                idx = groupings == g
                tile_mesh_sizes[idx] = np.min(tile_mesh_sizes[idx])
        if border_width is None:
            indx = self.overlaps
            bboxes = self.init_bboxes
            _, ovlp_wds = miscs.bbox_intersections(bboxes[indx[:,0]], bboxes[indx[:,1]])
            tile_border_widths = np.zeros(self.num_tiles, dtype=np.float32)
            np.maxium.at(tile_border_widths, indx, np.stack((ovlp_wds, ovlp_wds)), axis=-1)
            tile_border_widths = tile_border_widths / np.min(self.tile_sizes, axis=-1)
            # rounding to make mesh reuse more likely
            tile_border_widths = max(np.round(tile_border_widths/0.1), 1.0) * 0.1
            # tiles in a group share mesh
            grp_u, cnt = np.unique(groupings, return_counts=True)
            grp_u = grp_u[cnt>1]
            for g in grp_u:
                idx = groupings == g
                tile_border_widths[idx] = np.max(tile_border_widths[idx])
        elif hasattr(border_width,'__len__'):
            tile_border_widths = border_width
        else:
            tile_border_widths = np.full(self.num_tiles, border_width, dtype=np.float32)
        # make starting portion of the tile softer to make for distortion
        if soft_top != 1 and soft_top_width > 0:
            stf_y = interp1d([0, 0.99*soft_top_width, soft_top_width,1],
                             [soft_top, soft_top, 1, 1], kind='linear',
                             bounds_error=False, fill_value=(soft_top, 1))
        else:
            stf_y = None
        if soft_left != 1 and soft_left_width > 0:
            stf_x = interp1d([0, 0.99*soft_left_width, soft_left_width,1],
                             [soft_left, soft_left, 1, 1], kind='linear',
                             bounds_error=False, fill_value=(soft_left, 1))
        else:
            stf_x = None
        # soften the mesh with the strain from matching
        if (match_soften is None) or (match_soften[1] == 1):
            tile_soft_factors = np.ones(self.num_tiles, dtype=np.float32)
        else:
            strain_list = list(self.match_strains.items())
            overlap_indices = np.array([s[0] for s in strain_list])
            strain_vals = np.array([(s[1], s[1]) for s in strain_list])
            groupov_indices = groupings[overlap_indices]
            # only probe the interfaces between groups
            idxt = groupov_indices[:,0] != groupov_indices[:,1]
            groupov_indices = groupov_indices[idxt].ravel()
            strain_vals = strain_vals[idxt].ravel()
            avg_strain = np.zeros(self.num_tiles, dtype=np.float32)
            for g in np.unique(groupov_indices):
                idxt = groupov_indices == g
                strn = np.median(strain_vals[idxt])
                avg_strain[groupings == g] = strn
            upper_strain, lower_soft_factor = match_soften
            soften_func = interp1d([0, upper_strain], [1, lower_soft_factor],
                kind='linear', bounds_error=False, fill_value=(1, lower_soft_factor))
            tile_soft_factors = soften_func(avg_strain)
        meshes = []
        mesh_indx = np.full(self.num_tiles, -1)
        mesh_params_ptr = {} # map the parameters of the mesh to
        default_caches = {}
        for gear in MESH_GEARS:
            default_caches[gear] = defaultdict(lambda: miscs.CacheFIFO(maxlen=cache_size))
        self._default_mesh_cache = default_caches
        for k, tile_size in enumerate(self.tile_sizes):
            tmsz = tile_mesh_sizes[k]
            tbwd = tile_border_widths[k]
            tsf = tile_soft_factors[k]
            key = (*tile_size, tmsz, tbwd)
            if key in mesh_params_ptr:
                midx = mesh_params_ptr[key]
                mesh_indx[k] = midx
                M0 = meshes[midx].copy(override_dict={'uid':k, 'soft_factor':tsf})
            else:
                tileht, tilewd = tile_size
                M0 = Mesh.from_boarder_bbox((0,0,tilewd,tileht), bd_width=tbwd,
                    mesh_growth=interior_growth, mesh_size=tmsz, uid=k,
                    soft_factor=tsf)
                M0.set_stiffness_multiplier_from_interp(xinterp=stf_x, yinterp=stf_y)
                M0.center_meshes_w_offsets(gear=MESH_GEAR_FIXED)
                default_caches[key] = k
                mesh_indx[k] = k
            for gear in MESH_GEARS:
                M0.set_default_cache(cache=default_caches[gear], gear=gear)
            meshes.append(M0)
        for M, offset in zip(meshes, self._init_offset):
            M.apply_translation(offset, gear=MESH_GEAR_FIXED)
        self.meshes = meshes
        self.mesh_sharing = mesh_indx


    def initilize_optimizer(self, **kwargs):
        if (not kwargs.get('force_update', False)) and (self._optimizer is not None):
            return False
        if (self.meshes is None) or (self.num_links == 0):
            raise RuntimeError('meshes and matches not initialized for Stitcher.')
        self._optimizer = SLM(self.meshes, **kwargs)
        for key, match in self.matches.items():
            uid0, uid1 = key
            xy0, xy1, weight = match
            self._optimizer.add_link_from_coordinates(uid0, uid1, xy0, xy1,
                weight=weight, check_duplicates=False)
        return True


    def optimize_translation(self, **kwargs):
        """
        optimize the translation of tiles according to the matches for a specific
        gear.
        Kwargs:
            maxiter: maximum number of iterations in LSQR. None if no limit.
            tol: the stopping tolerance of the least-square iterations.
            start_gear: gear that associated with the vertices before applying
                the translation.
            target_gear: gear that associated with the vertices at the final
                postions for locked meshes. Also the results are saved to this
                gear as well.
            residue_threshold: if set, links with average error larger than this
                at the end of the optimization will be removed one at a time.

        """
        maxiter = kwargs.get('maxiter', None)
        tol = kwargs.get('tol', 1e-07)
        start_gear = kwargs.get('start_gear', MESH_GEAR_FIXED)
        target_gear = kwargs.get('target_gear', start_gear)
        residue_threshold = kwargs.get('residue_threshold', None)
        if self._optimizer is None:
            self.initilize_optimizer()
        self._optimizer.optimize_translation_lsqr(maxiter=maxiter, tol=tol,
            start_gear=start_gear, target_gear=target_gear)
        num_disabled = 0
        if (residue_threshold is not None) and (residue_threshold > 0):
            while True:
                lnks_w_large_dis = []
                mxdis = 0
                for k, lnk in enumerate(self._optimizer.links):
                    if not lnk.relevant():
                        continue
                    dxy = lnk.dxy(gear=(target_gear, target_gear), use_mask=True)
                    dxy_m = np.median(dxy, axis=0)
                    dis = np.sqrt(np.sum(dxy_m**2))
                    mxdis = max(mxdis, dis)
                    if dis > residue_threshold:
                        lnks_w_large_dis.append((dis, lnk.uids, k))
                if not lnks_w_large_dis:
                    break
                else:
                    lnks_w_large_dis.sort(reverse=True)
                    uid_record = set()
                    for lnk in lnks_w_large_dis:
                        dis, lnk_uids, lnk_k = lnk
                        if uid_record.isdisjoint(lnk_uids):
                            self._optimizer.links[lnk_k].disable()
                            num_disabled += 1
                        uid_record.update(lnk_uids)
                    self._optimizer.optimize_translation_lsqr(maxiter=maxiter,
                        tol=tol,start_gear=start_gear, target_gear=target_gear)
        return num_disabled


    def optimize_group_iterface(self, **kwargs):
        """
        initialize the mesh transformation based only on the matches between
        tiles from different groups. tiles within each group will have the same
        transformation.
        Kwargs: refer to the input of feabas.optimizer.SLM.optimize_linear.
        """
        if 'target_gear' in kwargs:
            target_gear = kwargs['target_gear']
        else:
            target_gear = MESH_GEAR_MOVING
            kwargs['target_gear'] = target_gear
        if not self.has_groupings:
            return
        groupings = self.groupings(normalize=True)
        match_uids = np.array(list(self.matches.keys()))
        match_grps = groupings[match_uids]
        idxt = match_grps[:,0] != match_grps[:,1]
        if not np.any(idxt):
            return
        match_uids = match_uids[idxt]
        sel_uids, sel_match_uids = np.unique(match_uids, return_inverse=True)
        sel_match_uids = sel_match_uids.reshape(-1,2)
        sel_grps = groupings[sel_uids]
        sel_meshes = [self.meshes[s].copy(override_dict={'uid':k}) for k,s in enumerate(sel_uids)]
        opt = SLM(sel_meshes)
        for uids0, uids1 in zip(match_uids, sel_match_uids):
            match = self.matches[tuple(uids0)]
            xy0, xy1, weight = match
            opt.add_link_from_coordinates(uids1[0], uids1[1], xy0, xy1,
                weight=weight, check_duplicates=False)
        opt.optimize_linear(groupings=sel_grps, **kwargs)
        for g, m in zip(groupings, self.meshes):
            sel_idx = np.nonzero(sel_grps == g)[0]
            if sel_idx.size == 0:
                continue
            else:
                m_border = opt.meshes[sel_idx[0]]
                v = m_border.vertices(gear=target_gear)
                m.set_vertices(v, target_gear)


    def optimize_elastic(self, **kwargs):
        """
        elastically optimize the spring mesh system.
        Kwargs:
            use_groupings(bool): whether to bundle the meshes within the same
                group when applying the transformation.
            residue_len(float): if set, the link weight will be adjusted
                according to the length before attempting another optimization
                routine.
            residue_mode: the mode to adjust residue. Could be 'huber' or
                'threshold'.
        other kwargs refer to the input of feabas.optimizer.SLM.optimize_linear.
        """
        use_groupings = kwargs.get('use_groupings', False) and self.has_groupings
        residue_len = kwargs.get('residue_len', 0)
        residue_mode = kwargs.get('residue_mode', None)
        if 'target_gear' in kwargs:
            target_gear = kwargs['target_gear']
        else:
            target_gear = MESH_GEAR_MOVING
            kwargs['target_gear'] = target_gear
        if use_groupings:
            groupings = self.groupings(normalize=True)
        else:
            groupings = None
        if self._optimizer is None:
            self.initilize_optimizer()
        cost = self._optimizer.optimize_linear(groupings=groupings, **kwargs)
        if (residue_mode is not None) and (residue_len > 0):
            if residue_mode == 'huber':
                self._optimizer.set_link_residue_huber(residue_len)
            else:
                self._optimizer.set_link_residue_threshold(residue_len)
            self._optimizer.adjust_link_weight_by_residue(gear=(target_gear, target_gear))
            cost1 = self._optimizer.optimize_linear(groupings=groupings, **kwargs)
            cost = [cost[0], cost1[-1]]
        return cost


    def connect_isolated_subsystem(self, **kwargs):
        """
        translate blocks of tiles that are connected by links as a whole so that
        disconnected blocks roughly maintain the initial relative positions.
        """
        gear = (MESH_GEAR_INITIAL, MESH_GEAR_MOVING)
        maxiter = kwargs.get('maxiter', None)
        tol = kwargs.get('tol', 1e-07)
        if self._optimizer is None:
            raise RuntimeError('optimizer of the stitcher not initialized.')
        Lbls, N_conn = self._optimizer.connected_subsystems
        if N_conn <= 1:
            return N_conn
        cnt = np.zeros(N_conn)
        np.add.at(cnt, Lbls, 1)
        overlaps = self.overlaps
        L_ov = Lbls[overlaps]
        idxt = L_ov[:,0] != L_ov[:,1]
        L_ov = L_ov[idxt]
        overlaps = overlaps[idxt]
        Neq = overlaps.shape[0]
        idx0 = np.stack((np.arange(Neq), np.arange(Neq)), axis=-1).ravel()
        idx1 = L_ov.ravel()
        v = (np.ones_like(L_ov) * np.array([1, -1])).ravel()
        idx0 = np.append(idx0, Neq)
        idx1 = np.append(idx1, np.argmax(cnt))
        v = np.append(v, 1)
        A = sparse.csr_matrix((v, (idx0, idx1)), shape=(Neq+1, N_conn))
        overlaps_u, overlaps_inverse = np.unique(overlaps, return_inverse=True)
        overlaps_inverse = overlaps_inverse.reshape(overlaps.shape)
        txy = np.array([self.meshes[s].estimate_translation(gear=gear) for s in overlaps_u])
        offset1 = txy[overlaps_inverse[:,0]] - txy[overlaps_inverse[:,1]]
        offset0 = self._init_offset[overlaps[:,0]] - self._init_offset[overlaps[:,1]]
        bxy = np.append(offset0 - offset1, [[0, 0]], axis=0)
        Tx = sparse.linalg.lsqr(A, bxy[:,0], atol=tol, btol=tol, iter_lim=maxiter)[0]
        Ty = sparse.linalg.lsqr(A, bxy[:,1], atol=tol, btol=tol, iter_lim=maxiter)[0]
        if np.any(Tx != 0) or np.any(Ty != 0):
            for m, lbl in zip(self.meshes, Lbls):
                tx = Tx[lbl]
                ty = Ty[lbl]
                m.unlock()
                m.apply_translation((tx, ty), gear[-1])
        # if still not fully connected, align average point
        A_blk = sparse.csr_matrix((np.ones_like(L_ov[:,0]), (L_ov[:,0], L_ov[:,1])), shape=(N_conn, N_conn))
        N_blk, L_blk = csgraph.connected_components(A_blk, directed=False, return_labels=True)
        if N_blk > 1:
            offset1 = np.array([m.estimate_translation(gear=gear) for m in self.meshes])
            dxy = self._init_offset - offset1
            Ls = L_blk[Lbls]
            for lbl in range(N_blk):
                txy = np.mean(dxy[Ls == lbl], axis = 0)
                for m, l in zip(self.meshes, Ls):
                    if l != lbl:
                        continue
                    m.unlock()
                    m.apply_translation(txy, gear[-1])
        return N_conn


    def normalize_coordinates(self, **kwargs):
        """
        apply uniform rigid transforms to the meshes so that the rotation is
        less than rotation_threshold and the upper-left corner is aligned to
        the offset.
        """
        gear = (MESH_GEAR_INITIAL, MESH_GEAR_MOVING)
        rotation_threshold = kwargs.get('rotation_threshold', None)
        offset = kwargs.get('offset', None)
        theta0 = np.nan
        offset0 = None
        if self._optimizer is None:
            raise RuntimeError('optimizer of the stitcher not initialized.')
        if rotation_threshold is not None:
            rotations = []
            for m in self.meshes:
                R = m.estimate_affine(gear=gear, svd_clip=(1,1))
                rotations.append(np.arctan2(R[0,1], R[0,0]))
            rotations = np.array(rotations, copy=False)
            L_ss, N_ss = self._optimizer.connected_subsystems
            theta = {}
            for lbl in range(N_ss):
                theta = np.median(rotations[L_ss == lbl])
                theta_d = theta * 180 / np.pi
                theta0 = max(theta0, theta_d)
                if theta > rotation_threshold:
                    c, s = np.cos(theta), np.sin(theta)
                    R = np.array([[c, s, 0], [-s, c, 0], [0, 0, 1]], dtype=np.float32)
                    for m, l in zip(self.meshes, L_ss):
                        if l != lbl:
                            continue
                        m.unlock()
                        m.apply_affine(R, gear[-1])
        if offset is not None:
            offset0 = np.array([np.inf, np.inf])
            for m in self.meshes:
                xy_min = m.bbox(gear=gear[-1], offsetting=True)[:2]
                offset0 = np.minimum(offset0, xy_min)
            txy = np.array(offset) - offset0
            if np.any(txy != 0):
                for m in self.meshes:
                    m.unlock()
                    m.apply_translation(txy, gear[-1])
        return theta0, offset0


    def set_groupings(self, groupings):
        """
        groupings can be used to indicate a subset of tiles should share the
        same mechanical properties and meshing options. Their fixed-pattern
        deformation will also be compensated identically (as necessary). This
        also means the tiles within the same group should share the same tile
        sizes.
        """
        if groupings is None:
            groupings = None
        else:
            groupings = np.array(groupings, copy=False)
            assert len(groupings) == len(self.imgrelpaths)
            tile_ht = self.tile_sizes[:,0]
            tile_wd = self.tile_sizes[:,1]
            grp_u, cnt = np.unique(groupings, return_counts=True)
            grp_u = grp_u[cnt > 1]
            if tile_ht.ptp() > 0:
                for g in grp_u:
                    idx = groupings == g
                    if tile_ht[idx].ptp() > 0:
                        raise RuntimeError(f'tile size in group {g} not consistent')
            if tile_wd.ptp() > 0:
                for g in grp_u:
                    idx = groupings == g
                    if tile_ht[idx].ptp() > 0:
                        raise RuntimeError(f'tile size in group {g} not consistent')
        self._groupings = groupings


    def groupings(self, normalize=False):
        if self._groupings is None:
            return np.arange(self.num_tiles)
        if normalize:
            if not hasattr(self, '_normalized_groupings') or self._normalized_groupings is None:
                _, indx = np.unique(self._groupings, return_inverse=True)
                self._normalized_groupings = indx
            return self._normalized_groupings
        else:
            return self._groupings


    @property
    def has_groupings(self):
        if self._groupings is None:
            return False
        groupings = self.groupings(normalize=True)
        if groupings.size == np.max(groupings):
            return False
        else:
            return True


    @property
    def overlaps_without_matches(self):
        overlaps = self.overlaps
        return np.array([s for s in overlaps if (tuple(s) not in self.matches)])


    @property
    def overlaps(self):
        if (not hasattr(self, '_overlaps')) or (self._overlaps is None):
            self._overlaps = self.find_overlaps()
        return self._overlaps


    def clear_mesh_cache(self, gear=None, instant_gc=True):
        if self._default_mesh_cache is not None:
            if gear is None:
                for g in MESH_GEARS:
                    self.clear_mesh_cache(gear=g)
            else:
                cache = self._default_mesh_cache[gear]
                if isinstance(cache, miscs.CacheNull):
                    cache.clear()
                elif isinstance(cache, dict):
                    for c in cache.values():
                        c.clear()
        if instant_gc:
            gc.collect()


    @property
    def num_tiles(self):
        return len(self.imgrelpaths)


    @property
    def num_links(self):
        return len(self.matches)


    @staticmethod
    def match_list_of_overlaps(overlaps, imgpaths, bboxes, **kwargs):
        """
        matching function used as the target function in Stitcher.dispatch_matchers.
        Args:
            overlaps(Kx2 ndarray): pairs of indices of the images to match.
            imgpaths(list of str): the paths to the image files.
            bboxes(Nx4 ndarray): the estimated bounding boxes of each image.
        Kwargs:
            root_dir(str): if provided, the paths in imgpaths are relative paths to
                this root directory.
            min_width: minimum amount of overlapping width (in pixels) to use.
                overlaps below this threshold will be skipped.
            index_mapper(N ndarray): the mapper to map the local image indices to
                the global ones, in case the imgpaths fed to this function is only
                a subset of all the images from the dispatcher.
            margin: extra image to include for matching around the overlapping
                regions to accommodate for inaccurate starting location. If
                smaller than 3, multiply it with the shorter edges of the
                overlapping bboxes before use.
            image_to_mask_path (list of str): this provide a way to input masks
                for matching. The file structure of the masks should mirror
                mirror that of the images, so that
                  imgpath.replace(image_to_mask_path[0], image_to_mask_path[1])
                direct to the mask of the corresponding images. In the mask,
                pixels with 0 values are not used. If no mask image are found,
                default to keep.
            loader_config(dict): key-word arguments passed to configure ImageLoader.
            matcher_config(dict): key-word arguments passed to configure matching.
        Return:
            matches(dict): M[(global_index0, global_index1)] = (xy0, xy1, conf).
            match_strains(dict): D[(global_index0, global_index1)] = strain.
        """
        root_dir = kwargs.get('root_dir', None)
        min_width = kwargs.get('min_width', 0)
        index_mapper = kwargs.get('index_mapper', None)
        margin = kwargs.get('margin', 1.0)
        image_to_mask_path = kwargs.get('image_to_mask_path', None)
        loader_config = kwargs.get('loader_config', {})
        matcher_config = kwargs.get('matcher_config', {})
        instant_gc = kwargs.get('instant_gc', False)
        margin_ratio_switch = 2
        if len(overlaps) == 0:
            return {}, {}
        bboxes_overlap, wds = miscs.bbox_intersections(bboxes[overlaps[:,0]], bboxes[overlaps[:,1]])
        if 'cache_border_margin' not in loader_config:
            overlap_wd0 = 6 * np.median(np.abs(wds - np.median(wds))) + np.median(wds) + 1
            if margin < margin_ratio_switch:
                loader_config['cache_border_margin'] = int(overlap_wd0 * (1 + margin))
            else:
                loader_config['cache_border_margin'] = int(overlap_wd0 + margin)
        image_loader = StaticImageLoader(imgpaths, bboxes, root_dir=root_dir, **loader_config)
        if image_to_mask_path is not None:
            mask_paths = [s.replace(image_to_mask_path[0], image_to_mask_path[1])
                for s in image_loader.filepaths_generator]
            mask_exist = np.array([os.path.isfile(s) for s in mask_paths])
            loader_config.update({'apply_CLAHE': False,
                                  'inverse': False,
                                  'number_of_channels': None,
                                  'preprocess': None})
            mask_loader = StaticImageLoader(mask_paths, bboxes, root_dir=None, **loader_config)
        else:
            mask_exist = np.zeros(len(imgpaths), dtype=bool)
        matches = {}
        strains = {}
        for indices, bbox_ov, wd in zip(overlaps, bboxes_overlap, wds):
            if wd <= min_width:
                continue
            if margin < margin_ratio_switch:
                real_margin = int(margin * wd)
            else:
                real_margin = int(margin)
            bbox_ov = miscs.bbox_enlarge(bbox_ov, real_margin)
            idx0, idx1 = indices
            bbox0 = bboxes[idx0]
            bbox1 = bboxes[idx1]
            bbox_ov0 = miscs.bbox_intersections(bbox_ov, bbox0)
            bbox_ov1 = miscs.bbox_intersections(bbox_ov, bbox1)
            img0 = image_loader.crop(bbox_ov0, idx0, return_index=False)
            img1 = image_loader.crop(bbox_ov1, idx1, return_index=False)
            if mask_exist[idx0]:
                mask0 = mask_loader.crop(bbox_ov0, idx0, return_index=False)
            else:
                mask0 = None
            if mask_exist[idx1]:
                mask1 = mask_loader.crop(bbox_ov1, idx1, return_index=False)
            else:
                mask1 = None
            weight, xy0, xy1, strain = stitching_matcher(img0, img1, mask0=mask0, mask1=mask1, **matcher_config)
            if xy0 is not None:
                continue
            offset0 = bbox_ov0[:2] - bbox0[:2]
            offset1 = bbox_ov1[:2] - bbox1[:2]
            xy0 = xy0 + offset0
            xy1 = xy1 + offset1
            if index_mapper is not None:
                idx0 = index_mapper[idx0]
                idx1 = index_mapper[idx1]
            matches[(idx0, idx1)] = (xy0, xy1, weight)
            strains[(idx0, idx1)] = strain
        if instant_gc:
            gc.collect()
        return matches, strains
