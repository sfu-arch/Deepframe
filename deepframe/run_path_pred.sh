#Author: Apala Guha

for line in `cat ${1}`
do
	
	app=`echo $line | cut -d / -f8`
	echo $app

	echo "hdfs dfs -rm -r -f /user/aguha/output*"
	hdfs dfs -rm -r -f /user/aguha/output*
	echo "hdfs dfs -rm -r -f /user/aguha/test*"
	hdfs dfs -rm -r -f /user/aguha/test*
	echo "hdfs dfs -rm -r -f /user/aguha/.sparkStaging/application*"
	hdfs dfs -rm -r -f /user/aguha/.sparkStaging/application*
	echo "hdfs dfs -rm -r -f /user/aguha/.Trash"
	hdfs dfs -rm -r -f /user/aguha/.Trash
	hdfs dfs -rm -r -f /user/aguha/${app}

	echo "scp aguha@cs-amoeba-n2.cs.surrey.sfu.ca:$line ."
	scp aguha@cs-amoeba-n2.cs.surrey.sfu.ca:$line ./path-profile-trace.gz

	rm -r sequence_ref_1
	scp -r aguha@cs-amoeba-n2.cs.surrey.sfu.ca:/home/aguha/path_pred/v23/${app}/sequence_ref_1 .
	cat sequence_ref_1/* > seq.txt
	

	spark-submit --master yarn --deploy-mode cluster --executor-memory 4G --driver-memory 8G --driver-cores 4 train.py file:///home/aguha/path_prediction/path-profile-trace.gz ${app} ${@:2}
	spark-submit --master yarn --deploy-mode cluster --executor-memory 4G --driver-memory 8G --driver-cores 4 validate.py file:///home/aguha/path_prediction/path-profile-trace.gz ${app} file:///home/aguha/path_prediction/seq.txt ${@:2}
	

	rm -rf result/${app}
	echo "hdfs dfs -get /user/aguha/"${app}" result"
	hdfs dfs -get /user/aguha/${app}  result
	echo "scp -r result/"${app}" aguha@cs-amoeba-n2.cs.surrey.sfu.ca:/home/aguha/path_pred/v24"
	scp -r result/${app} aguha@cs-amoeba-n2.cs.surrey.sfu.ca:/home/aguha/path_pred/v24
	hdfs dfs -rm -r -f /user/aguha/${app}



done

exit

#for running a small test

echo "hdfs dfs -rm -r -f /user/aguha/output*"
hdfs dfs -rm -r -f /user/aguha/output*
echo "hdfs dfs -rm -r -f /user/aguha/test*"
hdfs dfs -rm -r -f /user/aguha/test*
echo "hdfs dfs -rm -r -f /user/aguha/.sparkStaging/application*"
hdfs dfs -rm -r -f /user/aguha/.sparkStaging/application*
echo "hdfs dfs -rm -r -f /user/aguha/.Trash"
hdfs dfs -rm -r -f /user/aguha/.Trash

rm -r 164.gzip
app='164.gzip'

time spark-submit --master local[16] --deploy-mode client train.py file:///home/aguha/path_prediction/smallprofile.gz file:///home/aguha/path_prediction/164.gzip 8 test 2 1 2 1 2 2 1000 2000 2>&1 | tee result/$app.log
time spark-submit --master local[16] --deploy-mode client validate.py file:///home/aguha/path_prediction/smallprofile.gz file:///home/aguha/path_prediction/164.gzip file:///home/aguha/path_prediction/164.gzip.seq 8 test 2 1 2 1 2 2 1000 2000 test 0.5 2>&1 | tee result/$app.log
echo "hdfs dfs -cat /user/aguha/test*/* > result/test.out"
hdfs dfs -cat /user/aguha/test*/* > result/test.out
exit
























