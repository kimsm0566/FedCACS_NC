#!/bin/bash
set -e

# 로그 디렉토리 생성
mkdir -p ./save/log

# 오류 처리
trap 'echo "Error occurred in experiment $index with dataset: $dataset_name, algorithm: $alg" ' ERR

# 실험 설정 리스트
declare -a experiments=(
    "--dataset cifar10 --model cnn --num_users 100 --shard_per_user 5 --user_cat_list 5"
    # "--dataset cifar100 --model cnn --num_users 100 --shard_per_user 5 --user_cat_list 5"
    # "--dataset cifar100 --model cnn --num_users 100 --shard_per_user 10 --user_cat_list 10"
    # "--dataset mini_imagenet --model convnet --num_users 100 --shard_per_user 5 --user_cat_list 5 --cos 1 --num_classes 100 --novel_class_num 0"
    # "--dataset mini_imagenet --model convnet --num_users 100 --shard_per_user 10 --user_cat_list 10 --cos 1 --num_classes 100 --novel_class_num 0"
)

# 각 실험 설정에 대한 반복
for index in "${!experiments[@]}"; do
    experiment="${experiments[$index]}"
     # 파라미터 추출
    dataset_name=$(echo "$experiment" | grep -oP '(?<=--dataset\s)\w+')
    num_users=$(echo "$experiment" | grep -oP '(?<=--num_users\s)\w+')
    shard_per_user=$(echo "$experiment" | grep -oP '(?<=--shard_per_user\s)\w+')

    for alg in "fedcacsNC"; do # "fedcacs" 
        for nc in 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9; do
            echo "Running experiment ${index} with dataset: ${dataset_name}, algorithm: ${alg}..."
            # 로그 파일 이름 설정
            log_file="./save/log/${alg}_${dataset_name}_${num_users}_${shard_per_user}_epochs100_nc${nc}.log"

            # 실행 명령어 설정 (필수 파라미터 추가 및 재설정)
            command="nohup python -u main.py --alg ${alg} --frac 0.1 --local_bs 10 --lr 0.01 --nc ${nc} --epochs 100 --local_ep 10 --local_second_ep 5 --gpu 1 --multi_cats 1 ${experiment} > ${log_file} 2>&1 &"
        
            echo "Running command: $command"
            
            # 백그라운드 작업 실행 및 PID 저장
            eval "$command"
            pid=$!

            # 백그라운드 작업 완료 대기
            wait $pid
            
            sleep 10  # GPU 메모리 정리를 위한 대기 시간
            
            echo "Finished experiment ${index}: ${dataset_name}, algorithm: ${alg}."
        done
    done
done

echo "All experiments completed!"