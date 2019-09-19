input=$1
lookahead=$2
coverage=$3


#for app in 164.gzip  179.art  183.equake  401.bzip2  444.namd  456.hmmer  470.lbm  blackscholes  bodytrack  dwt53  fft-2d
#do
for app in blackscholes
do

	./sequence-app.sh ${input} ${lookahead} ${coverage} ${app} &


done
