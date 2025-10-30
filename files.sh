awk 'BEGIN{jpg=0; xml=0} 
     /\.jpg$/ {jpg++} 
     /\.xml$/ {xml++} 
     END{print "JPG:", jpg, "XML:", xml, "DIFF:", jpg - xml}' < <(
  find . -mindepth 2 -maxdepth 2 \( -iname '*.jpg' -o -iname '*.xml' \)
)

