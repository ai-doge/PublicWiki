hugo
pushd public
git pull origin main
git add .
git commit -am 'update'
git push origin main 
popd
