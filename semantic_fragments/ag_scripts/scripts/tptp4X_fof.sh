prob_dir=/vol/bitbucket/aeg19/logical_plms/data/TPTP-v7.4.0/Problems/PUZ
tptp4X_pth=/vol/bitbucket/aeg19/logical_plms/data/TPTP-v7.4.0/Scripts/tptp4X
outdir=/vol/bitbucket/aeg19/logical_plms/data/TPTP-v7.4.0/fof_problems

# files=$(ls $prob_dir)

for file in $prob_dir/PUZ001-1.p $prob_dir/PUZ001+2.p
do
    $tptp4X_pth -d $outdir -tfofify -V $file
done