# Dataset

- https://www.ddbj.nig.ac.jp/dta/index-e.html
- https://ddbj.nig.ac.jp/public/ddbj_database/dta/

Download dataset with

- python download_data.py

Get the traces folder of the scf files:

```bash
mkdir -p "extracted_scf" && for file in data/FLJ/scf/\*.tgz; do tar -xzvf "$file" -C "extracted_scf"; done
mkdir -p "extracted_fasta" && for file in data/FLJ/fasta/*.gz; do gunzip -c "$file" > "extracted_fasta/$(basename "$file" .gz)"; done
```

Then run the sanger.R script on the scf files in extracted_scf

```bash
pip3 install wandb
wandb login
```

# AWS EC2

- I make an EC2
- I have a key downloaded
- I sshed into it
  - ssh -i "test-key.pem" ec2-user@ec2-44-203-68-112.compute-1.amazonaws.com
- move files from my machine to that one

  - make requirements.txt with pipreqs

- wget https://ftp-trace.ncbi.nlm.nih.gov/sra/sdk/current/sratoolkit.current-ubuntu64.tar.gz
- tar -xzf sratoolkit.current-ubuntu64.tar.gz
- export PATH=$PATH:~/sratoolkit.3.0.6-ubuntu64/bin
- source ~/.bashrc
- vdb-config -i
  - https://www.youtube.com/watch?v=rjjrHnZfymU&list=PL7dF9e2qSW0ZZci13mHSKZYis3MV4Mdoa&index=1
  - just set up aws as your cloud provider and report cloud instance identity
- test whether it works
  - fasterq-dump SR5368359
