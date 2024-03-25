from sqlite3 import Blob
import firebase_admin
from firebase_admin import credentials, storage
from datetime import datetime, timedelta
import os


cred = credentials.Certificate("Firebase/serviceAccountKey.json")
firebase_admin.initialize_app(
    cred, {"storageBucket": "stock-price-prediction-74e2b.appspot.com"})

bucket = storage.bucket()


def uploadARIMAfile(file_path, today_date, quote):

    destination_blob_name = "ARIMA/"+today_date+"/"+quote
    blob = bucket.blob(destination_blob_name)
    blob.upload_from_filename(file_path)
    print("upload successfully in ARIMA")


def uploadLSTMfile(file_path, today_date, quote):
    destination_blob_name = "LSTM/"+today_date+"/"+quote
    blob = bucket.blob(destination_blob_name)
    blob.upload_from_filename(file_path)
    print("upload successfully in LSTM")


def uploadLINEARREGRESSIONfile(file_path, today_date, quote):
    destination_blob_name = "LINEAR REGRESSION/"+today_date+"/"+quote
    blob = bucket.blob(destination_blob_name)
    blob.upload_from_filename(file_path)
    print("upload successfully in LR")


def uploadTrendsFile(file_path, today_date, quote):
    destination_blob_name = "TRENDS/"+today_date+"/"+quote
    blob = bucket.blob(destination_blob_name)
    blob.upload_from_filename(file_path)
    print("trend image uploaded succesfully")




def getARIMAlink(quote, today_date):
    file_name = "ARIMA/"+today_date+"/"+quote

    blob = bucket.blob(file_name)
    file_url = blob.generate_signed_url(expiration=timedelta(days=100))
    return file_url


def getLSTMlink(quote, today_date):
    file_name = "LSTM/"+today_date+"/"+quote
    blob = bucket.blob(file_name)
    file_url = blob.generate_signed_url(expiration=timedelta(days=100))
    return file_url


def getLRlink(quote, today_date):
    file_name = "LINEAR REGRESSION/"+today_date+"/"+quote
    blob = bucket.blob(file_name)
    file_url = blob.generate_signed_url(expiration=timedelta(days=100))
    return file_url


def getTrendLink(quote, today_date):
    file_name = "TRENDS/"+today_date+"/"+quote
    blob = bucket.blob(file_name)
    file_url = blob.generate_signed_url(expiration=timedelta(days=100))
    return file_url


def getImageLinkFromFirebase(quote):
    today_date = datetime.today().strftime("%d-%m-%Y")
    arimalink = getARIMAlink(quote, today_date)
    lstmlink = getLSTMlink(quote, today_date)
    lrlink = getLRlink(quote, today_date)
    trendslink = getTrendLink(quote, today_date)
    return arimalink, lstmlink, lrlink, trendslink
