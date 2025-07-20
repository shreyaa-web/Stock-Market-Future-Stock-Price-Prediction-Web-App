# Stock-Market-Future-Stock-Price-Prediction-Web-App
Stock Market Future Prediction Web App based on Machine Learning and Sentiment Analysis of News (API keys included in code). The front end of the Web App is based on Flask and Wordpress. The App predicts stock prices of the next seven days for any given stock under NASDAQ or NSE as input by the user. Predictions are made using three algorithms: ARIMA, LSTM, Linear Regression. The Web App combines the predicted prices of the next seven days with the sentiment analysis of news and it gives the recommendation whether the price of the particular stock is going to rise or fall in the future.

## Wordpress files
Due to the space limit download the wordpress file from [here](https://drive.google.com/file/d/1y34yup3uSOcKr1Hhp-_RPp530cJbDXCZ/view?usp=sharing).

## Screenshots
![image](https://user-images.githubusercontent.com/76894348/175959454-738a9e26-dfed-4f7f-be5c-5770e5d3a8ac.png)
![image](https://user-images.githubusercontent.com/76894348/175959472-fb5e1d5b-28d1-45c0-be0c-ff3407b4a313.png)
![image](https://user-images.githubusercontent.com/76894348/175959488-7b2abd21-a058-4127-9d64-e9549c4eed77.png)
![image](https://user-images.githubusercontent.com/76894348/175959510-eed7be6a-cc3b-4d6e-9081-b047551a566c.png)
![image](https://user-images.githubusercontent.com/76894348/175959534-04c0897d-6328-429a-bd7a-0a3321c623ab.png)
![image](https://user-images.githubusercontent.com/76894348/175959550-33397e96-2797-4f0a-adfc-006f72ea5db5.png)
![image](https://user-images.githubusercontent.com/76894348/175959877-5fbeb7db-8e43-45e6-9bc0-75267b2b0230.png)
![screenshot (3)](https://user-images.githubusercontent.com/76894348/175959902-fbed2bb8-da05-4a71-a0f4-29dce1ecfd7c.png)

## Technologies Used

- Wordpress
- Flask
- Tensorflow
- Keras
- Alphavantage
- Scikit-Learn
- Python
- PHP
- CSS
- HTML
- Javascript

## Instructions on how to install and Use the project

- Clone the repo. Download and install XAMPP server from https://www.apachefriends.org/download.html and start Apache and MySql servers
- Open phpmyadmin by visiting http://localhost/phpmyadmin/ and go to User Accounts -> Add a User, give username and password as admin and click on Check All next to Global Privileges and hit Go
- Next, create a new database named wordpress
- Select the wordpress database and click on Import and select the wordpress.sql file from the repo.
- Download my wordpress website zip file from here
- Extract the above zip file in xampp/htdocs folder
- Go to command prompt, change directory to directory of repository and type pip install -r requirements.txt
- To run app, type in command prompt, set FLASK_APP=app press Enter then  python -m flask run and again enter
- Open your web browser and go to http://localhost/wordpress-5.9.3/wordpress and click on the wordpress folders to access the web app
- Wordpress Admin Panel is available at: http://localhost/wordpress-5.9.3/wordpress/wp-admin
