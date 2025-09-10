import pandas as pd
import matplotlib.pyplot as plt


df = pd.read_csv("Lumina_Dataset.csv")


plt.figure(figsize=(10,4))
plt.plot(df['EAR'], label='EAR')
plt.plot(df['PERCLOS'], label='PERCLOS')
plt.xlabel('Frame / Sample')
plt.ylabel('Value')
plt.title('Eye Features Over Time')
plt.legend()
plt.show()


plt.figure(figsize=(10,4))
plt.hist(df['YAWNS'], bins=2, rwidth=0.8)
plt.xlabel('Yawn')
plt.ylabel('Count')
plt.title('Yawn Count Distribution')
plt.show()

plt.figure(figsize=(10,4))
plt.hist(df['Pitch'], bins=20, rwidth=0.8) 
plt.xlabel('Pitch (Normalized)')
plt.ylabel('Number of Frames')
plt.title('Distribution of Head Pitch')
plt.show()




df = pd.read_csv("Lumina_Dataset.csv")
summary = df.describe()
print(summary)
