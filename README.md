# Docker container for serving up data from processed videos

## build the container
```docker build -f Dockerfile.flask -t facefront.flask .```

## run the container
```docker run -p 5000:5000 [-v /dirwithVideos:/mdata] facefront.flask``` . 
dirwithVideos - place where you have name_hash.ext video files for serving up frames.  

## Interacting with the container

### Endpoints



#### return a frame
```http://yourserver:5000/api/1.0/frames/<string:file_hash>/<int:frame_number>```  

  GET    
  Get a frame=`frame_number` from video which file contenthash = `file_hash`.  
  
```bash
curl localhost:5000/api/1.0/frames/012d28de1d13820b471cf00e9e3ecf4e/128
```
  
```javascript
{"meta": {"file_hash": "012d28de1d13820b471cf00e9e3ecf4e", 
          "frame_number": 128}, 
 "frame": "static/764346b3676fa262acb06753c116c923.jpg"}
```
  
#### list the feeds
```http://yourserver:5000/api/1.0/feeds```  
  GET  
  Get a listing of the feeds that are available, must have mapped videos into the /mdata directory.
  
```bash
  curl localhost:5000/api/1.0/feeds
```
  
```javascript
  {'meta': {'result_set': {'count': 3}},
 'results': [{'file_content_hash': '012d28de1d13820b471cf00e9e3ecf4e',
              'hash': '012d28de1d13820b471cf00e9e3ecf4e',
              'location': '/mdata/3_012d28de1d13820b471cf00e9e3ecf4e.mp4',
              'name': '3_012d28de1d13820b471cf00e9e3ecf4e.mp4',
              'uri': '/static/3_012d28de1d13820b471cf00e9e3ecf4e.mp4'},
             {'file_content_hash': '01f678d7122a2c64eef9c02cde82ef29',
              'hash': '01f678d7122a2c64eef9c02cde82ef29',
              'location': '/mdata/1_01f678d7122a2c64eef9c02cde82ef29.mp4',
              'name': '1_01f678d7122a2c64eef9c02cde82ef29.mp4',
              'uri': '/static/1_01f678d7122a2c64eef9c02cde82ef29.mp4'},
             {'file_content_hash': '67e1198a466f88d0172adc77abde5b69',
              'hash': '67e1198a466f88d0172adc77abde5b69',
              'location': '/mdata/2_67e1198a466f88d0172adc77abde5b69.mp4',
              'name': '2_67e1198a466f88d0172adc77abde5b69.mp4',
              'uri': '/static/2_67e1198a466f88d0172adc77abde5b69.mp4'}]}

```
#### is it working
```http://yourserver:5000/api/1.0/working```  
  GET  
  will probbaly converted to a health heartbeat when scheduling via marathon  
  
#### make an image query
  POST  
  Upload a picture of a face to find which media has that face.
  
```bash
curl -i -X POST -H "Content-Type: multipart/form-data" -F "threshold=0.35" -F "0=@/dir/to/file.jpg" http://localhost:5000/api/1.0/results/matches
```     

```javascript
{'meta': {'query': {'feeds': {'012d28de1d13820b471cf00e9e3ecf4e': {'name': 'BTTF3_012d28de1d13820b471cf00e9e3ecf4e.mp4'},
                              '01f678d7122a2c64eef9c02cde82ef29': {'name': 'BTTF1_01f678d7122a2c64eef9c02cde82ef29.mp4'},
                              '67e1198a466f88d0172adc77abde5b69': {'name': 'BTTF2_67e1198a466f88d0172adc77abde5b69.mp4'}},
                    'threshold': 0.465},
          'result_set': {'count': 3, 'matches': 65},
          'vector_set': {'count': 1,
                         'vectors': [{'face_coordinates': [74, 168, 152, 91],
                                      'face_pic_hash': '1c42c84874523a521e3c98ced69537a0',
                                      'hash': '1c42c84874523a521e3c98ced69537a0',
                                      'vector': [...]}]}},
 'results': [{'distance': 0.463400028360126,
              'hash': '2b35f54522c5fca0894a32bf69cc5ef0',
              'src': '1c42c84874523a521e3c98ced69537a0',
              'uri': 'static/2b35f54522c5fca0894a32bf69cc5ef0.jpg',
              'videos': [{'frames': [{'face_coordinates': [100.0,
                                                           113.0,
                                                           274.0,
                                                           243.0],
                                      'id': 88590}],
                          'hash': '01f678d7122a2c64eef9c02cde82ef29'},
                         {'frames': [{'face_coordinates': [128.0,
                                                           194.0,
                                                           278.0,
                                                           301.0],
                                      'id': 155430}],
                          'hash': '67e1198a466f88d0172adc77abde5b69'},
                         {'frames': [{'face_coordinates': [104.0,
                                                           396.0,
                                                           176.0,
                                                           451.0],
                                      'id': 170220},
                                     {'face_coordinates': [93, 472, 172, 415],
                                      'id': 170250}],
                          'hash': '012d28de1d13820b471cf00e9e3ecf4e'}]}]}
```

#### compare two faces
  POST
  upload 2 faces and get vector distance between them
   
```bash
curl -i -X POST -H "Content-Type: multipart/form-data" -F "0=@/path/to/face1.jpg"  -F "1=@/path/to/face2.jpg"  http://localhost:5000/api/1.0/results/comparisons
```

```javascript
{'meta': {'vector_set': {'count': 2,
                         'vectors': [{'face_coordinates': [74, 168, 152, 91],
                                      'face_pic_hash': '1c42c84874523a521e3c98ced69537a0',
                                      'hash': '1c42c84874523a521e3c98ced69537a0',
                                      'vector': [...]},
                                     {'face_coordinates': [74, 168, 152, 91],
                                      'face_pic_hash': '1c42c84874523a521e3c98ced69537a0',
                                      'hash': '1c42c84874523a521e3c98ced69537a0',
                                      'vector': [...]}]}},
 'results': {'distance': 0.0}}
```




