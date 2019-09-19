input=$1
lookahead=$2
coverage=$3
app=$4

cd ${app}

#copy the mined sequences

rm -r sequence_${input}_${lookahead}
cp -r /home/aguha/path_pred/v27/${app}/sequence_${input}_${lookahead} .
cat sequence_${input}_${lookahead}/* | sort -g -r -t, -k2,2 > frequent_sequences.txt


#sort in descending order by weight and process each line
counter=0
rm success-${lookahead}.log

echo "******"
direc=`dirname */epp-sequences.txt`
echo $direc



rm ${direc}/path-series-*-${lookahead}.txt
rm ${direc}/${app}-needle-*-${lookahead}
rm needle-run-seq-*-${lookahead}.log
rm needle-seq-*-${lookahead}.log

sum=0

while read -r seq
do

	counter=`expr "$counter" + 1`


	paths=`echo ${seq} | awk -F, '{print $1}'`
	echo "***********", ${paths}

	wt=`echo ${seq} | awk -F, '{print $2}' | sed -e 's/[eE]+*/\\*10\\^/'`



	rm path-seq.txt
	touch path-seq.txt

	#now get each path in sequence and search in epp-sequences file
	for ((number=1;number <= ${lookahead};number++))
	{
		#get path
		hexpath=`echo ${paths} | awk -v x=$number '{print toupper($x)}'`
		path=`echo "ibase=16; ${hexpath}" | bc`


		#search path in epp-sequences
		search=`echo "^${path} "`
		grep $search $direc/epp-sequences.txt >> path-seq.txt


	
	}



	#check that none of the paths in the sequence were un-acceleratable
	npaths=`wc -l path-seq.txt | awk '{print $1}'`
	echo $npaths 




	if [ $npaths -eq $lookahead ]
	then

		cat path-seq.txt


		cp path-seq.txt ${direc}/path-series-${counter}-${lookahead}.txt
		cp ${direc}/path-series-${counter}-${lookahead}.txt ${direc}/path-seq.txt

		


		make needle-run-path
		if [ $? -eq 0 ]
		then
			echo "NEEDLE success on ${counter}" >> success-${lookahead}.log
			cat needle-path.log needle-run-path.log >> success-${lookahead}.log
			sum=`echo "scale=5; ${sum}+${wt}" | bc`
			echo $sum
			flag=`echo "${sum} > ${coverage}" | bc` 
			if [ $flag -eq 1 ]
			then
				break
			fi
		else
			echo "NEEDLE failure on ${counter}" >> success-${lookahead}.log
			cat needle-path.log needle-run-path.log >> success-${lookahead}.log
		fi

		
		mv needle-run-path.log needle-run-seq-${counter}-${lookahead}.log
		mv needle-path.log needle-seq-${counter}-${lookahead}.log
		

		mv ${direc}/${app}-needle-0 ${direc}/${app}-needle-${counter}-${lookahead}

		#break
	fi
done < frequent_sequences.txt
cd ..





