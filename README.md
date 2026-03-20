age-wiz

Age guessing by image. Written with Python.

My images are off the path - I am a beautiful woman with privacy! Nevertheless, I'll add more features this week. I'll be adding a front end.

At this moment, I trained this with my 4 pictures from age 22 and 21. I only trained this to guess the person herself.

A year ago, my friend said: "I aged like a fine wine." (BRO:DDD)

Also 2 years ago I was loving someone else and a girl was loving him. She told me that "You are older than everyone in this school. You'll never find love."

Go your own way bitch! I am more beautiful, smarter, better than you in every field of the competition... Laughing my fucking ass off!

Okay, in prep I was not that pretty because I thought that I was in a school with smart people. So, I didn't care much about how I was looking - because I thought that our minds were beyond our beauty. I was wrong!

As I am tall, have pretty skin, pretty eyes and a pretty hair and a brilliant mind as a default, I only had to lose 22lbs and buy 2-3 makeup products and a few blouses to defeat you eternally from "flesh" marketplace.

So after computer science, I'll be in medical school and not a single person will be able to guess my age. Because I hacked my telomeres. Yes I am Deadpool. I am cancer, just like my tumors my other cells are recurring too!!!!!!!!!!!!! Henceforth, I will be sexy forever!!!!

## Technical Details
- Python-based age estimation
- Trained on personal dataset (4 images, ages 21-22)

## How to Use

### Setup
```bash
git clone https://github.com/nosey-dewdrop/age-wiz.git
cd age-wiz
pip install -r requirements.txt
```

> Note: `dlib` requires `cmake` to build. On macOS run `brew install cmake` first. On Ubuntu run `sudo apt install cmake`.

### Train with your own photos
Create folders under `training/` with the person's name. Put at least 2-3 clear face photos in each folder.
```
training/
  john_25/
    photo1.jpg
    photo2.jpg
  sarah_30/
    photo1.jpg
    photo2.jpg
```

Then train:
```bash
python detector.py --train
```

### Test
```bash
# test a single image
python detector.py --test path/to/photo.jpg

# test all images in the validation folder
python detector.py --validate
```

You can also use `--model cnn` for more accurate results (slower, GPU recommended).

## Future Implementations
Frontend
Age Guessing
