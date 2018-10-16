
:<<!EOF!
	Batch processing for training and testing:
	b: benchmark number
	train_p: ratio of selected labeled samples
	seed: random seed
	log_SSL: folder to store log files 
!EOF!


mkdir -p log_SSL

for b in 2; do
for train_p in 0.1; do 
for seed in 150; do 
    ini="iccad${b}_config.ini"

    python gen_ini.py ${b} ${train_p} ${seed} ${ini}

    log="log_SSL/SSL_train_p${train_p}_seed${seed}.${ini}.log"

    echo train_p = ${train_p}, seed = ${seed}, log = ${log}
    cat ${ini} > ${log}
    python ./train_SSL.py ${ini} >> ${log}
    python ./test_SSL.py ${ini} >> ${log}

done 
done
done
