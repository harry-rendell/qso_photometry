result_1.csv > ztf_oids.csv
for j in {2..6}; do
	tail -q -n +2 result_$j.csv >> ztf_oids.csv
done