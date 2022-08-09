# bash
export UID=$(id -u)
export GID=$(id -g)
docker build --build-arg USER=$USER \
             --build-arg UID=$UID \
             --build-arg GID=$GID \
             --build-arg PW=29653847 \
             -t masterarbeitkreilinger:1.0 \
             .
WANDB_API_KEY=b2b87b7a47a54d74a79cf8ceb131c26efe9418a5
WANDB_NAME="My first run"
WANDB_NOTES="Smaller learning rate, more regularization."
WANDB_ENTITY=kreilinger
