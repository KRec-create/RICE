python main.py --dataset=ml-1M --trainset=./dataset/ml-1M/train.txt --testset=./dataset/ml-1M/test.txt --model=LightGCN --num_neg=64 --margin=5.0 --loss=simce
python main.py --dataset=amazon-beauty --trainset=./dataset/amazon-beauty/train.txt --testset=./dataset/amazon-beauty/test.txt --model=LightGCN --num_neg=64 --margin=5.0 --loss=simce
python main.py --dataset=pinterest --trainset=./dataset/pinterest/train.txt --testset=./dataset/pinterest/test.txt --model=LightGCN --num_neg=64 --margin=10.0 --loss=simce
python main.py --dataset=amazon-book --trainset=./dataset/amazon-book/train.txt --testset=./dataset/amazon-book/test.txt --model=LightGCN --num_neg=64 --margin=10.0 --loss=simce
python main.py --dataset=amazon-cd --trainset=./dataset/amazon-cd/train.txt --testset=./dataset/amazon-cd/test.txt --model=LightGCN --num_neg=64 --margin=5.0 --loss=simce
python main.py --dataset=douban-book --trainset=./dataset/douban-book/train.txt --testset=./dataset/douban-book/test.txt --model=LightGCN --num_neg=64 --margin=10.0 --loss=simce





