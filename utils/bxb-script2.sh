#! /bin/sh

DB_PATH=$1
REPORT_PATH=$2
EXT_REF=$3
EXT_AI=$4
OUTPUT_NAME=$5
OUTPUT_NAME2=$6

OUTPUT="bxb"
cd $DB_PATH
echo "$DB_PATH"

rm "$OUTPUT".out sd.out
rm "$OUTPUT_NAME2".out
rm "$OUTPUT_NAME".out

for entry in *."$EXT_AI" ; do
      name=$(echo "$entry" | cut -f 1 -d '.')
      echo "bxb $name"
echo "output_2 $OUTPUT_NAME2"
      bxb -r $name -a "$EXT_REF" "$EXT_AI" -f "0" -L "$OUTPUT".out sd.out
      bxb -r $name -a "$EXT_REF" "$EXT_AI" -f "0" -S "$OUTPUT_NAME2".out
    done
sumstats "$OUTPUT".out >> "$OUTPUT_NAME".out

if [ ! -d "$REPORT_PATH" ]; then
    mkdir "$REPORT_PATH"
fi
mv "$OUTPUT_NAME".out "$REPORT_PATH"
mv "$OUTPUT_NAME2".out "$REPORT_PATH"
