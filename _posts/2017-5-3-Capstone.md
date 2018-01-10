---
layout: post
title: Predicting the Stock Market
date: 2017-12-01
purl: https://github.com/lcm98
published: true
categories: Project
---

Summary: Integrated financial information from the Alpha Vantage API into a neural network built using TensorFlow’s Python library to predict the S&P 500 closing stock price on the following day.

My data is mainly source via an API provided by the website AlphaVantage. I explored the options of using GoogleTrends data as well as Twitter data to complement that, but ended up deciding that it was not a good option for the scope of this project. 

Due to it's nature (purely numeric) there was not a huge need for complex transformations or feature engineering, rather the difficulties were in dealing with the timeseries nature.

Working with Stock data, what I found most important to decide, was whether the price of a stock will go up, or down. To this end I turned the focus of my project from a regression, where I attempted to predict the closing price, to a classification, where I try to predict if the price will change positively or negatively.

I used Tensorflow is that the granularity of control allows me to be generating regerssion values as well as classification predictions, while optimizing on the classification part of the problem, where if I used an sklearn regression it would optimize for regression metrics only.

To deploy this model, I envision it being set up on a server that is updated daily. Every day it can be used to generate the predicted closing price for that day (or the next depending on the time), at the end of the day once the true value is known, the Neural Net can update itself based on the now newly known data. Additionally, I imagine that a model of this type should eventually be built to work as a piece of a trading algorithim, with the model providing belief about what will happen in the future.

As for modeling techniques, I think it would be valuable to see how using a Bayesian approach would work for this kind of time series data. Additionally I would like to expand upon the idea of using a Neural Net to estimate both the Regression (change in price) and Classification (did price increase) parts of the problem, as right now I only have a very basic versino of that duality working.