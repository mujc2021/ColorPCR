if [ "$3" = "test" ]; then
    python test.py --test_epoch=$1 --benchmark=$2
fi
python eval.py --test_epoch=$1 --benchmark=$2 --method=lgr
 for n in 250 500 1000 2500 5000; do
     python eval_ori.py --test_epoch=$1 --num_corr=$n --benchmark=$2 --method=ransac
 done
