import pandas as pd
songs = {'Album' : ['Thriller', 'Back in Black', 'The Dark Side of the Moon',
                     'The Bodyguard', 'Bat Out of Hell'],
         'Released' : [1982,1982,1982,1992,1977],
         'Length' : ['00:42:19', '00:42:11', '00:42:49', '00:57:44', '00:46:33']}

songs_frame = pd.DataFrame(songs)
print(songs_frame)
print(" ")

unique_years = songs_frame['Released'].unique()
print(f"The unique years are {unique_years}")
print(" ")

df1 = songs_frame[songs_frame['Released']==1982]
print(df1)
print(" ")

print(songs_frame.iloc[0:2,0:3])

songs_frame.to_csv('songs_frame.csv')