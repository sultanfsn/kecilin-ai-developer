# Bakuten Shoot Corp Beyblade Battle Analytics
- [Installation](<#installation>)
- [How To Run](<#how-to-run>)

 Bakuten Shoot Corop Beyblade Analytics is a program intended for Kecilin take home assignment. This program has the ability to detect Beyblades and analyze some information from a battle video.

 ## Installation 
 create virtualenv
 ```
 pip install virtualenv
 python -m venv myenv
 myenv\Scripts\activate
 ```

 install requirements
 ```
 pip install -r requirements.txt
 ```
 create videos folder for input:
 ```
 mkdir videos
 ```

 ## How To Run
 This program is very simple to run. Set the input video name on main.py video_name.

 Run the program
 ```
 python main.py
 ```

 Your result will be stored in result folder. It will generate winner.jpg (image of winning beyblade), loser.jpg (image of losing beyblade), and battle_result.csv (battle summary datas).