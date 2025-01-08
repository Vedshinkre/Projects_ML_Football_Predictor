# Projects_Machine_Learning_Football_Predictor
## Why I Chose This Project 
Football has been a passion of mine since childhood. I grew up watching matches and idolizing FC Barcelona, immersing myself in the beautiful game and its endless intricacies. Analyzing teams, debating strategies, and predicting outcomes became second nature to me. 

When I began studying machine learning, I saw it as a perfect opportunity to combine my lifelong love for football with my growing interest in data science. Initially, my choice for this project was focused on La Liga, as it featured my favorite club, FC Barcelona. I later considered the Bundesliga due to its tactical diversity. However, I realized that these leagues didn’t exhibit the same level of unpredictability I was looking for. 

This led me to the English Premier League, particularly the mid-2010s seasons, renowned for their surprises and competitive balance. These years encapsulated the essence of unpredictability in football, making them an ideal dataset for a machine learning model. The idea of applying data science to predict outcomes in such a dynamic environment felt like an engaging way to learn and create something meaningful. 


## Project Overview 
### The project is divided into two main parts: 
#### 1. Scraping and Preparing Football Match Data 
 The first phase involves collecting historical match data from the English Premier League (EPL). Using Python and the requests library, I scrape match data for several seasons from an online source. 
After retrieving the raw data, I use BeautifulSoup to parse it and pandas to clean and organize it. The result is a consolidated pandas DataFrame containing all EPL matches for multiple seasons, ready for analysis. 

#### 2. Building a Predictive Model 
In the second phase, I process the cleaned EPL match data to create predictors (features) that the model will use to make predictions. These predictors might include metrics like goals scored, possession percentages, and other match statistics. 
I train a machine learning model to predict the outcome of each football match (home win, away win, or draw). For this, I use libraries like scikit-learn and implement techniques to measure the model's accuracy. 
Finally, I analyze the model's performance, identify areas for improvement, and optimize its predictions. 

## Installation
To follow this project, please install the following locally:

JupyerLab

Python 3.8+

Python packages
 
pandas

requests

BeautifulSoup

scikit-learn


