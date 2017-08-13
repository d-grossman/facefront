# Docker container for serving up data from processed videos

## build the container
```docker build -f Dockerfile.flask -t facefirst.flask .```

## run the container
```docker run -it -p 5000:5000 facefirst.flask /bin/bash```

## Interacting with the container

### Endpoints

#### is it working
  ```http://yourserver:5000/api/1.0/working``` .  
  GET . 
  will probbaly converted to a health heartbeat when scheduling via marathon . 
  
#### make_search_vector
  ```http://yourserver:5000/api/1.0/makevector``` . 
  POST . 
  post a file with a single face for getting search key . 

#### find_vectors
  ```http://yourserver:5000/api/1.0/find/<string:search_vector_name>/<float:distance>``` . 
  GET .  
  ```search_vector_name``` - hash from picture upload . 
  ```distance``` - [0,1] [tight, fuzzy] .  

### upload a picture with 1 face, get a hash 
```curl -i -X POST -H "Content-Type: multipart/form-data" -F "data=@static/face.jpg"``` 

### use the hash to search for others from an image
```curl http://yourserver:5000/api/1.0/find/94667607017ff1f597afa7da7be8592b/0.5```



