# bash
export UID=$(id -u)
export GID=$(id -g)
docker build --build-arg USER=$USER \
             --build-arg UID=$UID \
             --build-arg GID=$GID \
             --build-arg PW=29653847 \
             -t  masterarbeit_kreilinger_pre_rppg_and_wcd_train_test:1.0 \
             .


