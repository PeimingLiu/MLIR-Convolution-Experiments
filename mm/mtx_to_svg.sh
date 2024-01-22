
ms=$(ls *.mtx)
for MAT in $ms;
do

exec 5< $MAT
while read line <&5; do
  if [[ $line == %* ]]; then
    echo $line
  else
    break
  fi
done

crd=( $line )
echo "<svg width=\"${crd[0]}\" height=\"${crd[1]}\" version=\"1.1\" xmlns=\"http://www.w3.org/2000/svg\">" > ${MAT}.svg
while read line <&5; do
  crd=( $line )
  echo "<rect x=\"${crd[0]}\" y=\"${crd[1]}\" width=\"1\" height=\"1\"/>" >> ${MAT}.svg
done
echo "</svg>" >> ${MAT}.svg

done
