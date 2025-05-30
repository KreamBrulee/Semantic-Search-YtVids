# This is a sample Python script.
import csv
import urllib.request
import json
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
import os
from pinecone import Pinecone
from youtube_transcript_api import YouTubeTranscriptApi


def getAllVidsInPlaylist(playlist_id):
    YT_api_key = "User API Key Here"
    base_vid_url = "https://www.youtube.com/watch?v="
    base_playlist_url = "https://www.googleapis.com/youtube/v3/playlistItems?"

    url = base_playlist_url+"key={}&playlistId={}&part=snippet,id&order=date&maxResults=20".format(YT_api_key, playlist_id)

    data = urllib.request.urlopen(url)
    resp = json.load(data)
    vid_links = []

    for i in resp["items"]:
        if i["kind"] == "youtube#playlistItem":
            vid_links.append(i["snippet"]["resourceId"]["videoId"])

    trans=[]
    for id in vid_links:
        try:
            tran = YouTubeTranscriptApi.get_transcript(id)
            for a in tran:
                a.update({'url': base_vid_url + id})
                trans.append(a)
        except:
            print(f"Couldn't find trans for {base_vid_url+id}")

    print(trans)
    with open('transcriptses.csv', 'w') as f:
        # Write all the dictionary keys in a file with commas separated.
        f.write(','.join(trans[0].keys()))
        f.write('\n')  # Add a new line
        for row in trans:
            # Write the values in a row.
            f.write(','.join(str(x) for x in row.values()))
            f.write('\n')  # Add a new line

    return trans

def Embeder(line):

    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device='cpu')
    line.replace("\n"," ")
    return (model.encode(line))

def getEmbed(file):
    df = open(file,'r')
    reader = csv.reader(df)
    arm = list(reader)

    ds = pd.DataFrame(columns=['Embeding','Url','Timestamp'])
    err = 'a'
    i = 0
    try:
        for elm in arm:
            if i > 0:
                print(elm[0])
                emb = Embeder(elm[0])
                link = elm[3]
                time = elm[1]
                ds.loc[len(ds.index)] = [emb, link, time]
                print(i)
            i += 1
    except(err):
        ds.to_csv('trans_emb.csv')
        print(err)

    ds.to_csv('trans_emb.csv')

def upsert(file):
    pc = Pinecone(api_key="User API Key Here")
    index = pc.Index("markiplier-gtfo")

    df = open(file, 'r')
    reader = csv.reader(df)
    arm = list(reader)

    vals = []
    i=0
    for elm in arm:
        if i == 14999:
            vals.clear()

        if i == 15867:
            index.upsert(vals)
            vals.clear()
        if i>0:
            data =elm[1]
            data = data.replace("[",'')
            data = data.replace("]", '')
            dat = np.array(data.split(), dtype=np.float32)
            vals.append({'id':i.__str__(), 'values':dat, 'metadata':{"vidlink":elm[2], "start":elm[3], "timelink":elm[2]+'&t='+elm[3]}})
            i+=1
        elif i==0:
            i += 1
            continue
        else:
            break

    print(vals)

def qry():
        model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device='cpu')

        pc = Pinecone(api_key="245d6ad9-6cad-494e-8865-dca16a9dd33f")
        index = pc.Index("markiplier-gtfo")

        sentence=input("Querry?: ")
        emb_q=model.encode(sentence).tolist()

        #string = ','.join(str(x) for x in emb_q)
        #val = np.array(string.split(','), dtype=np.float32)
        result= index.query(vector=emb_q, top_k=5, include_metadate=True)

        temp = result['matches']

        identity=[]
        for x in temp:
            identity.append(x['id'])

        final = index.fetch(identity)['vectors']
        links=[]
        for x in identity:
            tl=final[str(x)]['metadata']['timelink'][0:-4]
            links.append(tl)

        print(links)



# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    #trans = getAllVidsInPlaylist("PL3tRBEVW0hiCLJHISDKnlPJ7Jr6fPmiHs")
    #getEmbed ("transcriptses.csv")
    #upsert ('trans_emb.csv')
    qry ()



