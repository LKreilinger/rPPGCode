# bash
export UID=$(id -u)
export GID=$(id -g)
docker build --build-arg USER=$USER \
             --build-arg UID=$UID \
             --build-arg GID=$GID \
             -t masterarbeitkreilinger:1.0 \
             .

wandb login b2b87b7a47a54d74a79cf8ceb131c26efe9418a5
