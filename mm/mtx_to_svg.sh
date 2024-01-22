
ms=$(ls *.mtx)
for MAT in $ms;
do

exec 5< $MAT
sym=false
while read line <&5; do

  if [[ $line == %* ]]; then
    if [[ $line == *symmetric* ]]; then
      sym=true
    fi
  else
    break
  fi
done

crd=( $line )
echo "<svg width=\"${crd[1]}\" height=\"${crd[0]}\" version=\"1.1\" xmlns=\"http://www.w3.org/2000/svg\">" > ${MAT}.svg
while read line <&5; do
  crd=( $line )
  echo "<rect x=\"${crd[1]}\" y=\"${crd[0]}\" width=\"1\" height=\"1\"/>" >> ${MAT}.svg

  if [[ $sym = true ]]; then
    echo "<rect x=\"${crd[0]}\" y=\"${crd[1]}\" width=\"1\" height=\"1\"/>" >> ${MAT}.svg
  fi
done
echo "</svg>" >> ${MAT}.svg

done
