# !/bin/bash 
# To declare static Array  

read -p "which cuda to use? (0/1)" cuda_idx
cuda_idx=${cuda_idx:-0}
read -p "Wanna evaluate which case (genome/medical):" scenario
scenario=${scenario:-medical}
# loops iterate through a  
# set of values until the 
# list (arr) is exhausted
trap '{ echo "Hey, you pressed Ctrl-C.  Time to quit." ; exit 1; }' INT


if [ "$scenario" == "genome" ]; then

arr=("bert" "gpt" "gpt-2" "xlm" "xlnet" "roberta" "ernie") 

echo "Evaluate GENOME Case"
for model in "${arr[@]}"
do
    # access each element  
    # as $i
    CUDA_VISIBLE_DEVICES=$cuda_idx python adv_genome_position.py -c --save_p default -a $model -t
done
fi


if [ "$scenario" == "medical" ]; then
    # arr=("bert" "gpt" "gpt2" "xlm" "xlnet" "roberta" "ernie")
    arr=("bert-base")
    read -p "Which attack model? (SVM/MLP/DANN)" atk
    atk=${atk:-DANN}
    echo "Evaluation MEDICAL Case with $atk"
    for model in "${arr[@]}"
    do
    # access each element  
    # as $i
	CUDA_VISIBLE_DEVICES=$cuda_idx python adv_YAN.py --clf $atk -a $model -v
	
    done
fi

exit 0
    
