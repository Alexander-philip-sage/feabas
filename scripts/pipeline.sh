#qsub -A BrainImagingML -q debug -l select=2 -l walltime=00:60:00 -k doe -l filesystems=home:eagle -N pipeline ./scripts/pipeline.sh
#works with spawn, failes with fork /usr/bin/time mpiexec --np 1 --ppn 1 --depth 32 --cpu-bind depth python3 /eagle/BrainImagingML/apsage/feabas/scripts/stitch_mpi.py --work_dir $WORKDIR  --mode all
#fails with fork /usr/bin/time mpiexec --np 1 --ppn 1 --depth 32  python3 /eagle/BrainImagingML/apsage/feabas/scripts/stitch_mpi.py --work_dir $WORKDIR  --mode all
#fails with fork /usr/bin/time mpiexec --np 1 --ppn 1 python3 /eagle/BrainImagingML/apsage/feabas/scripts/stitch_mpi.py --work_dir $WORKDIR  --mode all
#fails with fork /usr/bin/time mpiexec python3 /eagle/BrainImagingML/apsage/feabas/scripts/stitch_mpi.py --work_dir $WORKDIR  --mode all
#runs, just takes 15 min vs 0.7 min /usr/bin/time mpiexec --np 1 --ppn 16 python3 /eagle/BrainImagingML/apsage/feabas/scripts/stitch_mpi.py --work_dir $WORKDIR  --mode render
#14.6 min /usr/bin/time mpiexec --np 1 --ppn 16 --cpu-bind verbose,list:0,1:2,3:4,5:6,7:8,9:10,11:12,13:14,15:16,17:18,19:20,21:22,23:24,25:26,27:28,29:30,31 python3 /eagle/BrainImagingML/apsage/feabas/scripts/stitch_mpi.py --work_dir $WORKDIR  --mode render
#13.8 min /usr/bin/time mpiexec --np 1 --ppn 16 --envall --cpu-bind verbose,list:0,1:2,3:4,5:6,7:8,9:10,11:12,13:14,15:16,17:18,19:20,21:22,23:24,25:26,27:28,29:30,31 python3 /eagle/BrainImagingML/apsage/feabas/scripts/stitch_mpi.py --work_dir $WORKDIR  --mode render

. /eagle/BrainImagingML/apsage/feabas/env_polaris.sh 
MPIVERSION=$(mpiexec -V)
echo $MPIVERSION
WORKDIR='/eagle/BrainImagingML/apsage/feabas/work_dir4'
#rm -r $WORKDIR/stitched_sections
#rm -r $WORKDIR/stitch/match_h5
#rm -r $WORKDIR/stitch/tform
find . -type d -regex "./work_dir4/stitched_sections/mip[1-9]" -exec rm -rf {} \;
rm -r $WORKDIR/thumbnail_align
rm -r $WORKDIR/align/mesh
rm -r $WORKDIR/align/tform
rm -r $WORKDIR/aligned_stack
NNODES=`wc -l < $PBS_NODEFILE`
PPN=2
RANKS=$(( NNODES*PPN ))
echo nnodes $NNODES
echo ranks $RANKS 
##--depth 32 --cpu-bind depth
#START=`date +"%s"`
#/usr/bin/time mpiexec --np $RANKS --ppn $PPN --depth 32 --cpu-bind depth \
#    python3 /eagle/BrainImagingML/apsage/feabas/scripts/stitch_mpi.py \
#    --work_dir $WORKDIR \
#    --mode matching_optimize_render
#echo stitch.matching_optimize_render finished
#NOW=`date +"%s"`
#echo $((NOW - START)) seconds

START=`date +"%s"`
/usr/bin/time mpiexec -n $RANKS --ppn $PPN --depth 32 --cpu-bind depth \
    python3 /eagle/BrainImagingML/apsage/feabas/scripts/thumbnail_mpi.py \
    --work_dir $WORKDIR \
    --mode downsample_alignment
NOW=`date +"%s"`
echo thumbnail downsample_alignment finished
echo $((NOW - START)) seconds
#START=`date +"%s"`
#/usr/bin/time mpiexec -n $RANKS --ppn $PPN --depth 32 --cpu-bind depth \
#    python3 /eagle/BrainImagingML/apsage/feabas/scripts/thumbnail_mpi.py \
#    --work_dir $WORKDIR \
#    --mode alignment
#NOW=`date +"%s"`
#echo thumbnail alignment finished
#echo $((NOW - START)) seconds

##copy matches so that you don't have to run align's fine alignment
#DEST=$WORKDIR'/align/'
#SRC=$WORKDIR'/thumbnail_align/matches/'
#mkdir -p $DEST
#cp -r $SRC $DEST
#echo about to start align mpi
#sleep 5
#START=`date +"%s"`
#/usr/bin/time mpiexec -n $NNODES --ppn $PPN --depth 32 --cpu-bind depth \
#    python3 /eagle/BrainImagingML/apsage/feabas/scripts/align_mpi.py \
#    --work_dir $WORKDIR \
#    --mode meshing
#echo align meshing finished
#NOW=`date +"%s"`
#echo $((NOW - START)) seconds
#START=`date +"%s"`
#/usr/bin/time mpiexec -n $NNODES --ppn $PPN --depth 32 --cpu-bind depth \
#    python3 /eagle/BrainImagingML/apsage/feabas/scripts/align_mpi.py \
#    --work_dir $WORKDIR \
#    --mode optimization
#echo align optimization finished
#NOW=`date +"%s"`
#echo $((NOW - START)) seconds
#START=`date +"%s"`
#/usr/bin/time mpiexec -n $NNODES --ppn $PPN --depth 32 --cpu-bind depth \
#    python3 /eagle/BrainImagingML/apsage/feabas/scripts/align_mpi.py \
#    --work_dir $WORKDIR \
#    --mode render
#echo align render finished
#NOW=`date +"%s"`
#echo $((NOW - START)) seconds
#mkdir -p $WORKDIR/aligned_stack/tr10_tc10
#cp $WORKDIR/aligned_stack/mip0/*/*tr10-tc10.png $WORKDIR/aligned_stack/tr10_tc10/



###############ARCHIVE##################
#NNODES=`wc -l < $PBS_NODEFILE`
#PPN=16
#RANKS=$(( NNODES*PPN ))
#START=`date +"%s"`
#/usr/bin/time mpiexec --np $RANKS --ppn $PPN --envall --cpu-bind verbose,list:0,1:2,3:4,5:6,7:8,9:10,11:12,13:14,15:16,17:18,19:20,21:22,23:24,25:26,27:28,29:30,31 python3 /eagle/BrainImagingML/apsage/feabas/scripts/stitch_mpi.py --work_dir $WORKDIR  --mode render
#NOW=`date +"%s"`
#echo stich render
#echo $((NOW - START)) seconds
