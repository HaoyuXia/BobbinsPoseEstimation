#! /bin/sh
export LD_LIBRARY_PATH=/home/xia/OpenGR/build/install/lib/

SPACER="----------"

echo ${SPACER}
echo "Running Super4PCS"
time -p /home/xia/OpenGR/build/install/bin/Super4PCS -i /home/xia/bobbins/model02.ply /home/xia/bobbins/model01.ply -o 0 -n 5000 -d 0.01 -t 1000 -r /home/xia/bobbins/model01_super.obj -m /home/xia/bobbins/mat_result_super.txt

#echo ${SPACER}
#echo "Running 4PCS"
#time -p /home/xia/OpenGR/build/install/bin/Super4PCS -i ./model02.ply ./model01.ply -o 0 -n 5000 -d 0.01 -t 1000 -r model01_4pcs.ply -m mat_result.txt -x
