#!/bin/bash
if [[ "$1" = train ]]
then
    python frozen_lake.py
fi

# then
#     python frozen_lake.py
# else
#     python -c "import t4; t4.Package.install('aleksey/fashion-mnist-clf', registry='s3://alpha-quilt-storage', dest='.')"
#     cp aleksey/fashion-mnist-clf/clf.h5 clf.h5
#     rm -rf aleksey/
#     python -m flask run --host=0.0.0.0 --port=8080
# fi