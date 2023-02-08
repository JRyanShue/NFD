while read LINE
do
	echo $LINE
	pip install $LINE
done <requirements.txt
