import streamlit as st
from nba_api.stats.endpoints import shotchartdetail
from nba_api.stats.endpoints import playercareerstats
from nba_api.stats.static import players
from nba_api.stats.static import teams
from nba_api.stats.endpoints import leaguegamefinder
from nba_api.stats.endpoints import playbyplayv2
from nba_api.stats.endpoints import leagueplayerondetails
from nba_api.stats.endpoints import leaguestandingsv3
from pandasql import sqldf
import json
from gtts import gTTS
from PIL import Image, ImageFilter
import pandas as pd
import altair as alt
import matplotlib as mpl
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
import datetime
import re
import operator
import requests
from io import BytesIO
from autoregression import AR
import time
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

#Common objects for every function used
nba_teams = teams.get_teams()
    
# App Introduction
def introduction():
    st.markdown("<h1 style='text-align: center; color: lightblue;'>NBA Full Service App</h1>", unsafe_allow_html=True)
    st.caption("<h1 style='text-align: center;'>By Tobi Bui</h1>",unsafe_allow_html=True)
    st.markdown("<h1 style='text-align: left; color: red;'>Introduction about the Project</h1>", unsafe_allow_html=True)
    st.markdown("""The objective of my project is to understand how the NBA website works and do it in a more technical way but not focus on the aesthetic look of the website.
                I want to discover how scouters as well as coaches evaluate a player's quality to make the best tactics and training. Additionally, I want to help them have the best views of the players performances to make better
                decision. It's also built so that they can know what to do in the future about the player as well as teams status to improve in the future and reduce the weaknesses they have currently. Finally I want to learn new data 
                science tools and apply machine learning as well as data analytics to improve my coding skills.
""")
    st.markdown("<h1 style='text-align: left; color: red;'>Website System Flows</h1>", unsafe_allow_html=True)
    st.subheader("Introduction")
    st.write("This is the front page of the website presenting the purpose, website flows and mechanism usages")
    
    st.subheader("NBA Player Career Analysis")
    st.caption("Inputs")
    st.write("Inputs: There are 2 inputs: first name, last name")
    st.caption("Outputs")
    st.write("Output: There're 2 main parts: data analysis and machine learning algorithms. For the data analysis, take a look at yourself and see what happens")
    st.write("For the Machine Learning sections, I use 2 algorithms: Autoregressive and RNN.")
    st.caption("Machine Learning")
    st.write("For the Autoregressive, I build it from scratch. In statistics, econometrics and signal processing, an autoregressive (AR) model is a representation of a type of random process; as such, it is used to describe certain time-varying processes in nature, economics, behavior, etc. The autoregressive model specifies that the output variable depends linearly on its own previous values and on a stochastic term (an imperfectly predictable term); thus the model is in the form of a stochastic difference equation (or recurrence relation which should not be confused with differential equation). Together with the moving-average (MA) model, it is a special case and key component of the more general autoregressive–moving-average (ARMA) and autoregressive integrated moving average (ARIMA) models of time series, which have a more complicated stochastic structure; it is also a special case of the vector autoregressive model (VAR), which consists of a system of more than one interlocking stochastic difference equation in more than one evolving random variable. Contrary to the moving-average (MA) model, the autoregressive model is not always stationary as it may contain a unit root. The main purpose of this algorithm is to predict the chosen stats in future seasons as long as you wanted it to be. But the recommendation is 10 for accuracy")
    st.write("For the RNN, I import it using scikit-learn library from Python. A recurrent neural network (RNN) is a class of artificial neural networks where connections between nodes can create a cycle, allowing output from some nodes to affect subsequent input to the same nodes. This allows it to exhibit temporal dynamic behavior. Derived from feedforward neural networks, RNNs can use their internal state (memory) to process variable length sequences of inputs.This makes them applicable to tasks such as unsegmented, connected handwriting recognition or speech recognition. Recurrent neural networks are theoretically Turing complete and can run arbitrary programs to process arbitrary sequences of inputs. The term recurrent neural network is used to refer to the class of networks with an infinite impulse response, whereas convolutional neural network refers to the class of finite impulse response. Both classes of networks exhibit temporal dynamic behavior.[8] A finite impulse recurrent network is a directed acyclic graph that can be unrolled and replaced with a strictly feedforward neural network, while an infinite impulse recurrent network is a directed cyclic graph that can not be unrolled. The main purpose of this algorithm is to predict the chosen stats while comparing to current active players to see how he's performing in the future so it might take a long time to run. Additionally, this can only predict the next season")
    
    st.subheader("NBA Team Seasonal Analysis")
    st.caption("Inputs")
    st.write("Inputs: There are 2 inputs: team name, the team's AI drawn image filter")
    st.caption("Outputs")
    st.write("Output: There're 2 main parts: team seasonal and team monthly data analysis. These are just basic analytic based on the up to date API that I crawled from the NBA API along with the image of that team designed in any filters you like.")
    
    st.subheader("NBA Podcast")
    st.caption("Inputs")
    st.write("Inputs: There are 2 inputs: team name, opposite team name")
    st.caption("Outputs")
    st.write("Output: There're 2 main parts: Matchup History Report and machine learning algorithms. For the Matchup History Report, take a look at yourself and see what happens")
    st.write("For the Machine Learning, I use the Random Forest model. This time, I implement the decision trees by myself and use the most relevant data columns to use as a affecting variable for the possible output of the playoff prediction if they ever face. Random forest is a commonly-used machine learning algorithm trademarked by Leo Breiman and Adele Cutler, which combines the output of multiple decision trees to reach a single result. Its ease of use and flexibility have fueled its adoption, as it handles both classification and regression problems.")
    
#NBA Player Career Analysis
def main_page():
    st.markdown("<h1 style='text-align: left; color: red;'>Data Analysis</h1>", unsafe_allow_html=True)
    
    # Creates a node for each player. It is important to note that players. get_player () does not work in this case
    first_name = st.text_input('First name')
    last_name = st.text_input('Last name')
    player_id = []
    player_first_name = []
    player_last_name = []
    player_active_status = []
    nba_players = players.get_players()
    
    #st.write('Number of players fetched: {}'.format(len(nba_players)))
    
    # Add player names and active status to the list of players.
    for i in nba_players:
        # Add player s first and last name to the list of players.
        if (not first_name or first_name.lower() in i['full_name'].lower()) and (not last_name or last_name.lower() in i['full_name'].lower()):
            player_first_name.append(i['first_name'])
            player_last_name.append(i['last_name'])
            player_active_status.append(i['is_active'])
            player_id.append(i['id'])
            
    # Print the players information and their career stats
    if first_name or last_name:
        # Declare global variable to use in the entire function
        player_data_stats = pd.DataFrame()
        
        # Split the printing result into 2 sides
        left, right = st.columns(2)
        
        # The left side prints the players name and active status
        with left:
            df = pd.DataFrame(list(zip(player_first_name, player_last_name,player_active_status)), columns =['First Name', 'Last Name', "Active Status"])
            st.write(df)
        
        # The right side prints the players career stats
        with right:
            # Catch the data for players career
            @st.cache
            def get_career_stats(player_id):
                career = playercareerstats.PlayerCareerStats(player_id=str(player_id))
                return career.get_data_frames()[0]
            
            # Warning for the users to view stats of 1 player at a time to avoid confusion
            st.warning('You should choose 1 player at a time to avoid confusion in the data analysis part')
                
            # This function writes the career stats for each player.
            selected_players = []
            for i in range(len(player_id)):
                full_name = player_first_name[i] + " " + player_last_name[i]
                
                #Checkbox function to print out the selected players
                show = st.checkbox(full_name)
                
                # Show the players stats.
                if show: 
                    selected_players.append(player_id[i])
                    player_data_stats = get_career_stats(player_id[i])
                    st.write(player_data_stats)
        
        # Space down to create space
        st.write("\n")
        
        # Split the printing results to 2 sides
        left1, right1 = st.columns(2)
        
        for player_id in selected_players:
            with left1:
                # A combination of bar and line chart to visualize Game Play and Game Start
                GP_dataframe = pd.DataFrame(list(zip(get_career_stats(player_id)['SEASON_ID'], get_career_stats(player_id)['GP'], get_career_stats(player_id)['GS'])), columns =['Season', 'Game Play', "Game Start"])
                base = alt.Chart(GP_dataframe).encode(x='Season:O')
                bar = base.mark_bar().encode(y='Game Play:Q')
                line =  base.mark_line(color='red').encode(
                    y='Game Start:Q'
                )
                game_play = (bar + line).properties(width=600, title='Game Play and Game Start by Season')
                st.altair_chart(game_play, use_container_width=True)
                
                # A line chart to visualize Field Goal Made and Field Goal Attempt
                FG_dataframe = pd.DataFrame(list(zip(get_career_stats(player_id)['SEASON_ID'], get_career_stats(player_id)['FGM'], get_career_stats(player_id)['FGA'])), columns =['Season', 'Field Goal Made', "Field Goal Attempt"])
                fig, ax = plt.subplots(figsize=(12,5))
                ax2 = ax.twinx()
                ax.set_title('Field Goal Analysis')
                ax.set_xlabel('Season')
                ax.plot(FG_dataframe['Season'], FG_dataframe['Field Goal Made'], color='green', marker='x')
                ax2.plot(FG_dataframe['Season'], FG_dataframe['Field Goal Attempt'], color='red', marker='o')
                ax.set_ylabel('Field Goal Made')
                ax2.set_ylabel('Field Goal Attempt')
                ax.legend(['Field Goal Made'])
                ax2.legend(['Field Goal Attempt'], loc='upper center')
                ax.yaxis.grid(color='lightgray', linestyle='dashed')
                plt.tight_layout()
                st.pyplot(fig)
            
            with right1:
                # A combination of area and line chart to visualize Field Goal 3 Made and Field Goal 3 Attempt
                FG3_dataframe = pd.DataFrame(list(zip(get_career_stats(player_id)['SEASON_ID'], get_career_stats(player_id)['FG3M'], get_career_stats(player_id)['FG3A'])), columns =['Season', '3-Point Field Goal Made', "3-Point Field Goal Attempt"])
                base1 = alt.Chart(FG3_dataframe).encode(x='Season:O')
                area = base1.mark_area().encode(y='3-Point Field Goal Made:Q')

                line1 =  base1.mark_line(color='red').encode(
                    y='3-Point Field Goal Attempt:Q'
                )
                three_point = (area + line1).properties(width=600, title='3-Point Field Goal Made and 3-Point Field Goal Attempt by Season')
                st.altair_chart(three_point, use_container_width=True)
                
                # A combination of scatter and line chart to visualize Free Throw Made and Free Throw Attempt
                FT_dataframe = pd.DataFrame(list(zip(get_career_stats(player_id)['SEASON_ID'], get_career_stats(player_id)['FTM'], get_career_stats(player_id)['FTA'])), columns =['Season', 'Free Throw Made', "Free Throw Attempt"])
                base2 = alt.Chart(FT_dataframe).encode(x='Season:O')
                scatter = base2.mark_circle().encode(y='Free Throw Made:Q')

                line2 =  base2.mark_line(color='red').encode(
                    y='Free Throw Attempt:Q'
                )
                free_throw = (scatter + line2).properties(width=600, title='Free Throw Made and Free Throw Attempt by Season')
                st.altair_chart(free_throw, use_container_width=True)
            
            # Create a 3x2 grid of line chart subplots of Rebound, Assist, Turn Over, Steal, Block, Personal Foul, Total Points, Total Minutes Played and Field Goal Percentage 
            fig, axs = plt.subplots(nrows=3, ncols=3, figsize=(30, 20))

            # Plot data on each subplot
            # Plot data on Rebound
            axs[0, 0].plot(get_career_stats(player_id)['SEASON_ID'], get_career_stats(player_id)['REB'])
            axs[0, 0].set_title('Rebound')
            axs[0, 0].tick_params(axis='x', labelrotation=90)  # rotate x-axis labels by 90 degrees
            
            # Plot data on Assist
            axs[0, 1].plot(get_career_stats(player_id)['SEASON_ID'], get_career_stats(player_id)['AST'])
            axs[0, 1].set_title('Assist')
            axs[0, 1].tick_params(axis='x', labelrotation=90)  # rotate x-axis labels by 90 degrees

            # Plot data on Turnover
            axs[0, 2].plot(get_career_stats(player_id)['SEASON_ID'], get_career_stats(player_id)['TOV'])
            axs[0, 2].set_title('Turn Over')
            axs[0, 2].tick_params(axis='x', labelrotation=90)  # rotate x-axis labels by 90 degrees
            
            # Plot data on Steal
            axs[1, 0].plot(get_career_stats(player_id)['SEASON_ID'], get_career_stats(player_id)['STL'])
            axs[1, 0].set_title('Steal')
            axs[1, 0].tick_params(axis='x', labelrotation=90)  # rotate x-axis labels by 90 degrees

            # Plot data on Block
            axs[1, 1].plot(get_career_stats(player_id)['SEASON_ID'], get_career_stats(player_id)['BLK'])
            axs[1, 1].set_title('Block')
            axs[1, 1].tick_params(axis='x', labelrotation=90)  # rotate x-axis labels by 90 degrees
            
            # Plot data on Personal Foul
            axs[1, 2].plot(get_career_stats(player_id)['SEASON_ID'], get_career_stats(player_id)['PF'])
            axs[1, 2].set_title('Personal Foul')
            axs[1, 2].tick_params(axis='x', labelrotation=90)  # rotate x-axis labels by 90 degrees
            
            # Plot data on Total Points
            axs[2, 0].plot(get_career_stats(player_id)['SEASON_ID'], get_career_stats(player_id)['PTS'])
            axs[2, 0].set_title('Total Points')
            axs[2, 0].tick_params(axis='x', labelrotation=90)  # rotate x-axis labels by 90 degrees
            
            # Plot data on Total Minutes Played
            axs[2, 1].plot(get_career_stats(player_id)['SEASON_ID'], get_career_stats(player_id)['MIN'])
            axs[2, 1].set_title('Total Minute Played')
            axs[2, 1].tick_params(axis='x', labelrotation=90)  # rotate x-axis labels by 90 degrees
            
            # Plot data on Field Goal Percentage
            axs[2, 2].plot(get_career_stats(player_id)['SEASON_ID'], get_career_stats(player_id)['FG_PCT'])
            axs[2, 2].set_title('Field Goal Percentage')
            axs[2, 2].tick_params(axis='x', labelrotation=90)  # rotate x-axis labels by 90 degrees
            
            # Set overall title for the plot
            fig.suptitle('General Statistics', fontsize=16)
            st.set_option('deprecation.showPyplotGlobalUse', False)
            st.pyplot()
            
            #Collect the player career shot data
            shot_json = shotchartdetail.ShotChartDetail(
                            team_id = get_career_stats(player_id)['TEAM_ID'],
                            player_id = player_id,
                            context_measure_simple = 'FGA',
                            season_nullable = None)
            
            # Load data into a Python dictionary
            shot_data = json.loads(shot_json.get_json())

            # Get the relevant data from our dictionary
            relevant_data = shot_data['resultSets'][0]

            # Get the headers and row data
            headers = relevant_data['headers']
            rows = relevant_data['rowSet']
            
            # There's no data for shot locations
            if rows == []:
                st.warning("Currently the NBA API doesn't update the data for shot visualization")
            
            # There's data for shot locations
            else:
                # Create pandas DataFrame for shot data collected
                player_data = pd.DataFrame(rows)
                player_data.columns = headers
                
                # Function to draw basketball court
                def create_court(ax, color):
                    
                    # Short corner 3PT lines
                    ax.plot([-220, -220], [0, 140], linewidth=2, color=color)
                    ax.plot([220, 220], [0, 140], linewidth=2, color=color)
                    
                    # 3PT Arc
                    ax.add_artist(mpl.patches.Arc((0, 140), 440, 315, theta1=0, theta2=180, facecolor='none', edgecolor=color, lw=2))

                    # Lane and Key
                    ax.plot([-80, -80], [0, 190], linewidth=2, color=color)
                    ax.plot([80, 80], [0, 190], linewidth=2, color=color)
                    ax.plot([-60, -60], [0, 190], linewidth=2, color=color)
                    ax.plot([60, 60], [0, 190], linewidth=2, color=color)
                    ax.plot([-80, 80], [190, 190], linewidth=2, color=color)
                    ax.add_artist(mpl.patches.Circle((0, 190), 60, facecolor='none', edgecolor=color, lw=2))

                    # Rim
                    ax.add_artist(mpl.patches.Circle((0, 60), 15, facecolor='none', edgecolor=color, lw=2))
                        
                    # Backboard
                    ax.plot([-30, 30], [40, 40], linewidth=2, color=color)
                    
                    # Remove ticks
                    ax.set_xticks([])
                    ax.set_yticks([])
                        
                    # Set axis limits
                    ax.set_xlim(-250, 250)
                    ax.set_ylim(0, 470)
                    
                    # General plot parameters
                    mpl.rcParams['font.family'] = 'Avenir'
                    mpl.rcParams['font.size'] = 18
                    mpl.rcParams['axes.linewidth'] = 2
                    
                    #Return result
                    return ax
                
                # Create basketball court plot
                fig, ax = plt.subplots(figsize=(4, 3.76))
                ax = create_court(ax, 'black')
                
                # Plot hexbin of shots
                ax.hexbin(player_data['LOC_X'], player_data['LOC_Y'] + 60, gridsize=(30, 30), extent=(-300, 300, 0, 940), cmap='Blues', zorder=0)
                
                # Plot hexbin of shots with logarithmic binning
                ax.hexbin(player_data['LOC_X'], player_data['LOC_Y'] + 60, gridsize=(30, 30), extent=(-300, 300, 0, 940), bins='log', cmap='Blues', zorder =0)
                
                # Annotate player name and season
                ax.text(0, 1.05, 'Shot Attempt Visualization', transform=ax.transAxes, ha='left', va='baseline')
                
                # Draw the figure for shots
                st.pyplot(fig)
        
        # Machine Learning to predict player's performance in the future
        st.markdown("<h1 style='text-align: left; color: red;'>Machine Learning</h1>", unsafe_allow_html=True)
        
        # Autoregression model
        st.subheader("Autoregression Model to predict every attribute of your selected player")
        
        # Choose the desire stats to predict
        prediction_variable = st.selectbox('Pick one', ['REB', 'AST','STL','BLK','TOV','PF','GP','GS','MIN','OREB','DREB'])
        
        # Choose the next season to predict
        season_prediction = st.number_input('Pick the next year season to predict', 1, 10)
        
        #Autoregression Algorithm
        def autoreg(prediction_variable,season_prediction):
            # Extract the time series for a particular player
            time_series = player_data_stats[prediction_variable].values
            
            # Initialize the AR model with order p=3
            model = AR(p = 10)
            
            # Fit the model to the time series data
            model.fit(time_series)
            
            # Generate future predictions for the next 10 time steps
            predictions = model.predict(time_series, num_predictions=season_prediction, mc_depth=10)
            
            # convert the predictions to a pandas dataframe
            predictions_df = pd.DataFrame(predictions, columns=[str(prediction_variable)])
            
            #Create the season_list based on the number of years you want to predict
            season_list = []
            for i in range(1,season_prediction+1):
                season_year = str(2022 + i) + '-' + str(2022 + i + 1)
                season_list.append(season_year)
            
            #Add the season_list to the prediction dataframe
            season_series = pd.Series(season_list)
            predictions_df['Season'] = season_series.values
            
            # Extract the predicted values
            predicted_values = predictions_df[prediction_variable].values
            
            # Compute the mean squared error
            mse = mean_squared_error(time_series[0:len(predicted_values)], predicted_values) 
            
            # Print the predictions
            return predictions_df, mse
        
        # Choose the future season to predict
        if season_prediction > 10:
            st.warning("Please select the season range less than 10 to make the prediction to be more accurate")
        else:
            # Get the start time
            start_time = time.time()
            
            # Check if the user chose their player
            if player_data_stats.empty:
                st.warning("You didn't select the players to do prediction")
            else:
                # Call the function and print the result
                predictions_df, mse = autoreg(prediction_variable, season_prediction)
                st.write(predictions_df)
            
                # Get the end time
                end_time = time.time()
                
                # Calculate the runtime in seconds
                runtime = end_time - start_time
                
                #The accuracy of this model
                st.write(f"MSE for {prediction_variable} predictions for player {player_id}: {mse}")
                
                # Display the runtime
                st.write(f"The Runtime it takes to make the prediction is: {runtime:.2f} seconds")
        
        # RNN model to predict future stats for next season
        st.subheader("RNN Model to predict every attribute of your selected player and compare to other players")
        
        # Check if the user chose their player
        if player_data_stats.empty:
                st.warning("You didn't select the players to do prediction")
        else:
            # Checkbox function for the user to choose the RNN because it takes about 10mins to run based on my Computational Power
            run_rnn = st.checkbox("Run RNN Machine Learning Model to compare your player's chosen stats vs other players")
            
            # If the user want to use RNN to predict
            if run_rnn:
                # Warn the user of the run time
                st.warning("This RNN kodel takes about 10 mins to run so be patient")
                
                # Choose the desire stats to predict
                rnn_prediction_variable = st.selectbox('Pick one', ['REB', 'AST','STL','BLK','TOV','PF','GP','GS','MIN','OREB','DREB'])
                
                # Load the dataset
                player_stats = pd.read_csv(r"C:\Users\Admin\nba\large_df.csv")

                # Convert categorical variables to one-hot encoding
                player_stats = pd.get_dummies(player_stats, columns=['TEAM_ABBREVIATION'])

                # Split the dataset into training and testing sets
                train_data, test_data = train_test_split(player_stats, test_size=0.2)

                # Split the training and testing data into features and labels
                train_features = train_data.drop(['SEASON_ID', 'GP','GS','MIN','FG_PCT','FG3_PCT','FT_PCT', 'AST','STL','BLK','TOV','PF', 'PTS', 'REB'], axis=1)
                train_labels = train_data[[rnn_prediction_variable]].values
                test_features = test_data.drop(['SEASON_ID', 'GP','GS','MIN','FG_PCT','FG3_PCT','FT_PCT', 'AST','STL','BLK','TOV','PF', 'PTS', 'REB'], axis=1)
                test_labels = test_data[[rnn_prediction_variable]].values

                # Build the RNN model
                model = Sequential()
                model.add(LSTM(units=50, return_sequences=True, input_shape=(train_features.shape[1], 1)))
                model.add(Dropout(0.2))
                model.add(LSTM(units=50, return_sequences=True))
                model.add(Dropout(0.2))
                model.add(LSTM(units=50))
                model.add(Dropout(0.2))
                model.add(Dense(units=1))

                # Compile the model
                model.compile(optimizer='adam', loss='mean_squared_error')

                # Train the model
                model.fit(train_features.values.reshape(-1, train_features.shape[1], 1), train_labels, epochs=100, batch_size=32)

                # Test the model
                predictions = model.predict(test_features.values.reshape(-1, test_features.shape[1], 1))

                # Compare predictions to actual labels
                comparison = pd.DataFrame({'Prediction': predictions.flatten(), 'Actual': test_labels.flatten()})
                
                # Print the result in dataframe format
                st.write(comparison)

# NBA Team Seasonal Analysis
def page2():
    # Data Analysis of the team stats
    st.markdown("<h1 style='text-align: left; color: red;'>Data Analysis</h1>", unsafe_allow_html=True)
    
    # Choose the team name
    team_name = st.text_input('Team Name')
    
    # Declare NBA Team AI drawn Images
    boston_image = Image.open("Boston Celtics.jpg")
    buck_image = Image.open("Bucks.jpg")
    hornet_image = Image.open("Charlotte Hornets.jpg")
    bulls_image = Image.open("Chicago Bulls.jpg")
    cav_image = Image.open("Cleveland Cavaliers.jpg")
    dallas_image = Image.open("Dallas Mavericks.jpg")
    nugget_image = Image.open("Denver Nuggets.jpg")
    piston_image = Image.open("Detroit Pistons.jpg")
    warrior_image = Image.open("Golden State Warriors.jpg")
    rocket_image = Image.open("Houston Rockets.jpg")
    pacer_image = Image.open("Indiana Pacers.jpg")
    kings_image = Image.open("Kings.jpg")
    knicks_image = Image.open("Knicks.jpg")
    clipper_image = Image.open("LA Clippers.jpg")
    laker_image = Image.open("Lakers.jpg")
    memphis_image = Image.open("Memphis.jpg")
    heat_image = Image.open("Miami Heat.jpg")
    nets_image = Image.open("Nets.jpg")
    pelicans_image = Image.open("New Orlean Pelicans.jpg")
    okc_image = Image.open("OKC.jpg")
    magic_image = Image.open("Orlando Magic.jpg")
    suns_image = Image.open("Phoenix Suns.jpg")
    portland_image = Image.open("Portland.jpg")
    sixer_image = Image.open("Sixers.jpg")
    spur_image = Image.open("Spurs.jpg")
    timberwolves_image = Image.open("Timberwolves.jpg")
    raptor_image = Image.open("Toronto Raptors.jpg")
    utah_image = Image.open("Utah Jazz.jpg")
    wizard_image = Image.open("Washington Wizards.jpg")
    hawks_image = Image.open("hawks.jpg")
    
    
    # After the user put the team name, begin collecting the team id, full name, nickname, city, state and year founded
    if team_name:
        # Declare list of team information to collect
        team_id = []
        team_full_name = []
        team_abbreviation_name = []
        team_nickname = []
        team_city = []
        team_state = []
        team_year_founded = []
        
        # Collect the team id, full name, nickname, city, state and year founded
        for i in nba_teams:
            if (not team_name or team_name.lower() in i['full_name'].lower()) or (not team_name or team_name.lower() in i['abbreviation'].lower()):
                team_full_name.append(i['full_name'])
                team_abbreviation_name.append(i['abbreviation'])
                team_nickname.append(i['nickname'])
                team_city.append(i['city'])
                team_state.append(i['state'])
                team_year_founded.append(i['year_founded'])
                team_id.append(i["id"])
        
        # Query for games where the chosen team was playing
        gamefinder = leaguegamefinder.LeagueGameFinder(team_id_nullable=team_id[0])
        
        # Put the result into dataframe
        games = gamefinder.get_data_frames()[0]

        # Clean the dataframe
        def modify_dataframe(game_season, indicator):
            game_season['GAME_DATE'] = pd.to_datetime(game_season['GAME_DATE'])
            if indicator ==0:
                game_season['Year'] = game_season['GAME_DATE'].dt.year
            else:
                game_season['Year'] = game_season['GAME_DATE'].dt.month
            game_season = game_season.drop(['GAME_DATE', 'GAME_ID'], axis=1)
            
            return game_season
        
        # function to analyze the stats, which is used for both seasonal and monthly analysis
        def team_analysis(game_season, indicator):
            # Call the cleaning data function
            game_season = modify_dataframe(game_season, indicator)
            
            # Calculate the average stats of the team in seasons
            game_season_win = sqldf("select year, count(WL) as WIN from game_season where WL = 'W' group by year")
            game_season_lose = sqldf("select year, count(WL) as LOSE from game_season where WL = 'L' group by year")
            game_season_average = sqldf('select year, round(avg(MIN)) as MIN, round(avg(PTS)) as PTS, round(avg(FGM)) as FGM, round(avg(FGA)) as FGA, avg(FG_PCT), round(avg(FG3M)) as FG3M, round(avg(FG3A)) as FG3A, avg(FG3_PCT), round(avg(FTM)) as FTM, round(avg(FTA)) as FTA, avg(FT_PCT), round(avg(OREB)) as OREB, round(avg(DREB)) as DREB, round(avg(REB)) as REB, round(avg(AST)) as AST, round(avg(STL)) as STL, round(avg(BLK)) as BLK, round(avg(TOV)) as TOV, round(avg(PF)) as PF from game_season GROUP BY Year')
            game_season_stats = pd.merge(pd.merge(game_season_win,game_season_lose,on='Year'),game_season_average,on='Year')

            # Reformat the seasons to apply in the dataframe and calculate total games played
            total_games = games.groupby(games.SEASON_ID.str[-4:])[['GAME_ID']].count().loc['1983':]
            for i in range(len(total_games)):
                season_year = str(1983 + i) + '-' + str(1983 + i + 1)
                total_games = total_games.rename(index={str(1983 + i): season_year})
            
            # Print the information dataframe
            df = pd.DataFrame(list(zip(team_full_name, team_abbreviation_name,team_nickname,team_city,team_state,team_year_founded)), columns =['Full Name', 'Abbreviation', "Nickname", "City", "State", "Year Founded"])
            st.write(df)
            
            # Split the results into 2 sides but add a middle variable to make distance for better visualization
            left, middle, right = st.columns(3)
            
            # Print the average seasonal stats
            with left:
                st.write(game_season_stats)
            
            # Print total games played by season
            with right:
                st.write(total_games)
            
            # Stacked bar chart to print Number of wins and loses by year and month
            win_lose = pd.melt(game_season_stats, id_vars=['Year'], value_vars=['WIN', 'LOSE'], var_name='Group', value_name='Value')
            win_lose_chart = alt.Chart(win_lose).mark_bar().encode(
                x=alt.X('Year:N', title='Year'),
                y=alt.Y('Value:Q', title='Number of games'),
                color=alt.Color('Group:N', legend=alt.Legend(title="Win/Lose"), scale=alt.Scale(range=['#1f77b4', '#ff7f0e']))
            ).properties(title="Number of Wins and Losses by Year")
            st.altair_chart(win_lose_chart)
            
            # Basic histogram to visualize points score range
            fig = px.histogram(game_season_stats, x="PTS")
            st.plotly_chart(fig, use_container_width=True)
            
            # Advanced scatter plot with functions to visualize the Field Goal Attempt
            field_goal = px.scatter(game_season_stats, x="FGM", y="FGA", animation_frame="Year", animation_group = "Year",
                                    log_x=False, size_max=55, range_x=[0,100], range_y=[0,100])
            st.plotly_chart(field_goal, use_container_width=True)
            
            # Histogram with optional bar numbers for Minutes Played
            graph = st.plotly_chart(go.Figure(layout_title_text="Minutes Played"))
            size = st.slider("Number of bars:", 2, 10, 4)
            def update_bar_chart(size):
                fig = go.Figure(
                    data=[go.Histogram(x=game_season_stats['MIN'], nbinsx=size)],
                    layout_title_text="Minutes Played"
                )
                graph.plotly_chart(fig)
            update_bar_chart(size)
            
            # Stacked bar chart with legends and symbols to indicate Field Goal 3 Made and Field Goal 3 Attempt 
            three_point_field_goal = px.bar(game_season_stats, x="Year", y=["FG3M", "FG3A"], barmode="stack", color_discrete_map={"FG3M": "green", "FG3A": "blue"}, pattern_shape_sequence=[".", "x", "+"])
            st.plotly_chart(three_point_field_goal, use_container_width=True)
            
            # 3D Graph to visualize average Free Throw Percentage
            free_throw = px.scatter_3d(game_season_stats, x="FTM", y="FTA", z="avg(FT_PCT)", color="Year",
                        color_discrete_map = {"FTM": "blue", "FTA": "green", "avg(FT_PCT)":"red"})
            st.plotly_chart(free_throw, use_container_width=True)
            
            # Create a 3x3 grid of line chart subplots
            fig1, axs1 = plt.subplots(nrows=3, ncols=3, figsize=(30, 20))

            # Plot data on each subplot
            # Plot data on Rebound
            axs1[0, 0].plot(game_season_stats['Year'], game_season_stats['REB'])
            axs1[0, 0].set_title('Rebound')
            axs1[0, 0].tick_params(axis='x', labelrotation=90)  # rotate x-axis labels by 90 degrees
            
            # Plot data on Assist
            axs1[0, 1].plot(game_season_stats['Year'], game_season_stats['AST'])
            axs1[0, 1].set_title('Assist')
            axs1[0, 1].tick_params(axis='x', labelrotation=90)  # rotate x-axis labels by 90 degrees

            # Plot data on Turnover
            axs1[0, 2].plot(game_season_stats['Year'], game_season_stats['TOV'])
            axs1[0, 2].set_title('Turn Over')
            axs1[0, 2].tick_params(axis='x', labelrotation=90)  # rotate x-axis labels by 90 degrees
            
            # Plot data on Steal
            axs1[1, 0].plot(game_season_stats['Year'], game_season_stats['STL'])
            axs1[1, 0].set_title('Steal')
            axs1[1, 0].tick_params(axis='x', labelrotation=90)  # rotate x-axis labels by 90 degrees

            # Plot data on Block
            axs1[1, 1].plot(game_season_stats['Year'], game_season_stats['BLK'])
            axs1[1, 1].set_title('Block')
            axs1[1, 1].tick_params(axis='x', labelrotation=90)  # rotate x-axis labels by 90 degrees
            
            # Plot data on Personal Foul
            axs1[1, 2].plot(game_season_stats['Year'], game_season_stats['PF'])
            axs1[1, 2].set_title('Personal Foul')
            axs1[1, 2].tick_params(axis='x', labelrotation=90)  # rotate x-axis labels by 90 degrees
            
            # Plot data on Average Field Goal Percentage
            axs1[2, 0].plot(game_season_stats['Year'], game_season_stats['avg(FG_PCT)'])
            axs1[2, 0].set_title('Average Field Goal Percentage')
            axs1[2, 0].tick_params(axis='x', labelrotation=90)  # rotate x-axis labels by 90 degrees
            
            # Plot data on Average Field Goal 3 Percentage
            axs1[2, 1].plot(game_season_stats['Year'], game_season_stats['avg(FG3_PCT)'])
            axs1[2, 1].set_title('Average 3-point Field Goal Percentage')
            axs1[2, 1].tick_params(axis='x', labelrotation=90)  # rotate x-axis labels by 90 degrees
            
            # Plot data on Offensive Rebound
            axs1[2, 2].plot(game_season_stats['Year'], game_season_stats['OREB'])
            axs1[2, 2].set_title('Offensive Rebound')
            axs1[2, 2].tick_params(axis='x', labelrotation=90)  # rotate x-axis labels by 90 degrees

            # Set overall title for the plot
            fig1.suptitle('General Statistics', fontsize=16)
            st.set_option('deprecation.showPyplotGlobalUse', False)
            st.pyplot()
        
        # Side bar checkbox function to view stats by season
        by_season = st.sidebar.checkbox("View stats by seasons", key='by_season')

        # Only display the visalization of 1 selected box
        if "by_month" not in st.session_state:
            st.session_state.by_month = False
        # Disable the other checkbox when this one is selected
        if st.session_state.by_month and by_season:
            st.session_state.by_month = False
            st.sidebar.markdown("**Note:** Only one option can be selected at a time.")
            
        # Run the analysis when this checkbox is selected
        # If choose by season
        if by_season:
            game_season = games.copy()
            indicator = 0
            team_analysis(game_season, indicator)
        
        # Side bar checkbox function to view stats by months
        by_month = st.sidebar.checkbox("View stats by months of 1 season", key='by_month')

        # Disable the other checkbox when this one is selected
        if st.session_state.by_season and by_month:
            st.session_state.by_season = False
            st.sidebar.markdown("**Note:** Only one option can be selected at a time.")

        # Run the analysis when this checkbox is selected
        # If choose by months
        if by_month:
            game_year = sqldf("SELECT DISTINCT SUBSTR(SEASON_ID, -4) AS year FROM games")
            year_list = game_year['year'].tolist()
            year_list = list(map(int, year_list))
            year_pick = st.slider('Pick one', year_list[len(year_list)-1], year_list[0])
            game_each_season = games[games.SEASON_ID.str[-4:] == str(year_pick)]
            indicator = 1
            team_analysis(game_each_season, indicator)
    
        if team_full_name[0] == "Boston Celtics":
            image = boston_image
        elif team_full_name[0] == "Milwaukee Bucks":
            image = buck_image
        elif team_full_name[0] == "Charlotte Hornets":
            image = hornet_image
        elif team_full_name[0] == "Chicago Bulls":
            image = bulls_image
        elif team_full_name[0] == "Cleveland Cavaliers":
            image = cav_image
        elif team_full_name[0] == "Dallas Mavericks":
            image = dallas_image
        elif team_full_name[0] == "Denver Nuggets":
            image = nugget_image
        elif team_full_name[0] == "Detroit Pistons":
            image = piston_image
        elif team_full_name[0] == "Golden State Warriors":
            image = warrior_image
        elif team_full_name[0] == "Atlanta Hawks":
            image = hawks_image
        elif team_full_name[0] == "Houston Rockets":
            image = rocket_image
        elif team_full_name[0] == "Indiana Pacers":
            image = pacer_image
        elif team_full_name[0] == "Sacramento Kings":
            image = kings_image
        elif team_full_name[0] == "New York Knicks":
            image = knicks_image
        elif team_full_name[0] == "Los Angeles Clipper":

            image = clipper_image
        elif team_full_name[0] == "Los Angeles Lakers":
            image = laker_image
        elif team_full_name[0] == "Memphis Grizzlies":
            image = memphis_image
        elif team_full_name[0] == "Miami Heat":
            image = heat_image
        elif team_full_name[0] == "Brooklyn Nets":
            image = nets_image
        elif team_full_name[0] == "New Orlean Pelicans":
            image = pelicans_image
        elif team_full_name[0] == "Oklahoma City Thunder":
            image = okc_image
        elif team_full_name[0] == "Orlando Magic":
            image = magic_image
        elif team_full_name[0] == "Phoenix Suns":
            image = suns_image
        elif team_full_name[0] == "Portland Trail Blazers":
            image = portland_image
        elif team_full_name[0] == "Philadelphia 76ers":
            image = sixer_image
        elif team_full_name[0] == "San Antonio Spurs":
            image = spur_image
        elif team_full_name[0] == "Minnesota Timberwolves":
            image = timberwolves_image
        elif team_full_name[0] == "Toronto Raptors":
            image = raptor_image
        elif team_full_name[0] == "Utah Jazz":
            image = utah_image
        elif team_full_name[0] == "Washington Wizards":
            image = wizard_image
        
        filters=st.selectbox("Choose your filter",options=["None","Blur","Contour","Emboss","Find Edges"])
        if filters=="None":
            pass
        elif filters=="Blur":
            image=image.filter(ImageFilter.BLUR)
        elif filters=="Contour":
            image=image.filter(ImageFilter.CONTOUR)
        elif filters=="Emboss":
            image=image.filter(ImageFilter.EMBOSS)
        elif filters=="Find Edges":
            image=image.filter(ImageFilter.FIND_EDGES)
        
        st.write(f"Here's the AI generated image of your favorite team with chosen filters: {team_full_name[0]}")
        st.image(image)
            
# NBA Podcast and Playoff Win Prediction 
def page3():
    st.markdown("<h1 style='text-align: left; color: red;'>Gamelog</h1>", unsafe_allow_html=True)
    
    # Input team name
    team_name = st.text_input('Team Name')
    
    # Input Opposite team name
    opposite_team_name = st.text_input('Opposing Team Name')
    
    # After the user finish inputting the team names
    if team_name and opposite_team_name:
        # Choose the date you want to see the stats
        match_date = st.date_input(
                        "Your Match Date",
                        datetime.date(2019, 7, 6))
    
        # Team search Information including team_id, nickname and full name function
        def search_team_info(name):
            # Declare team information lists
            team_id = []
            team_abbreviation_name = []
            team_full_name = []
            
            # Add team id, nickname and full name to the list
            for i in nba_teams:
                if (not name or name.lower() in i['full_name'].lower()) or (not name or name.lower() in i['abbreviation'].lower()):
                    team_id.append(i["id"])
                    team_abbreviation_name.append(i['abbreviation'])
                    team_full_name.append(i['full_name'])
            
            # Return the result
            return team_id, team_abbreviation_name, team_full_name
        
        # Query for games where the teams were playing
        gamefinder = leaguegamefinder.LeagueGameFinder(team_id_nullable=search_team_info(name=team_name)[0][0])
        
        # The first DataFrame of those returned is what we want.
        games = gamefinder.get_data_frames()[0]
        
        # The second DataFrame of those returned is what we want for oppsite team
        games = games[games.MATCHUP.str.contains(search_team_info(name=opposite_team_name)[1][0])]
        
        # Matchup History
        st.subheader("Matchup History:")
        st.write(games)
        
        # Print the stats report for selected date
        games1 = games.loc[(games['GAME_DATE'] == str(match_date))]
        
        #Check if the user enters the correct date
        if games1.empty:
            st.warning("You need to choose the correct date which can be found in the Matchup History Dataframe I provided below")
        else:
            st.subheader("The matchup result you looked for:")
            st.write(games1)
        
            # Get the selected date game ID
            games_id = games1.GAME_ID
            
            # Get the selected date game dataframe
            pbp = playbyplayv2.PlayByPlayV2(games_id)
            pbp = pbp.get_data_frames()[0]
            
            # Copy the result to another fucntion
            df = pbp.copy()
            
            # Replace NA value with empty space
            pbp = pbp.fillna('')
            
            # Format for the Score column 
            pbp['SCORE'] = pbp.apply(lambda x: 'The score is now ' + str(x['SCORE'] +'.') if x['SCORE'] != "" else "", axis=1)
            
            # Create a function to get the Score report based on the dataframe and create a new Score column with detailed description to put in the dataframe
            def get_score_string(row):
                # Declare new variable list  to gather necessary information such as score, scoremargin
                score = row['SCORE']
                score_margin = row['SCOREMARGIN']
                team_abbreviation = ''
                
                # When the score is not empty
                if score != ' ':
                    # Score status: TIE
                    if isinstance(score_margin, str) and score_margin == 'TIE':
                        team_abbreviation = 'The score is tie'
                    
                    # Score status: LEADS for the home team
                    elif score_margin != '' and int(score_margin) > 0:
                        team_abbreviation = search_team_info(name=team_name)[2][0] + " " + "leads "
                    
                    # Score status: LEADS for the opposite team
                    elif score_margin != '' and int(score_margin) < 0:
                        team_abbreviation = search_team_info(name=opposite_team_name)[2][0] + " " + "leads "
                    
                    # Return the result of scores
                    return score + ' ' + team_abbreviation

                # When the score is empty
                else:
                    return ''
                
            # Get the lattest score report
            pbp['SCORE'] = pbp.apply(get_score_string, axis=1)
            
            # Update the final result of the game in the last column
            last_row = pbp.iloc[-1]
            last_row_score = last_row['SCORE']
            if 'leads' in last_row_score:
                last_row_score = last_row_score.replace('leads', 'wins')
            pbp.at[pbp.index[-1], 'SCORE'] = last_row_score

            pbp = pbp.assign(ColumnA = pbp.HOMEDESCRIPTION.astype(str) + \
                pbp.NEUTRALDESCRIPTION.astype(str) + pbp.VISITORDESCRIPTION.astype(str) \
                + pbp.SCORE.astype(str) + '. ')
            
            # Extract the comments from the original dataframe
            df = pbp.ColumnA
            sum = ''
            for index, value in df.items():
                sum += value
            
            # Create a autonomous voice recap for listening when you are too lazy to read the report: This can take about 5 mins to finish
            voice_recap = st.checkbox("Hear recap of this match")
            
            # Warning the time to process when the user want to hear the report
            st.warning('This might take about a minute to load because the recap file is large')
            
            # If they choose to hear, print the recording
            if voice_recap:
                sound_file = BytesIO()
                tts = gTTS(sum, lang='en')
                tts.write_to_fp(sound_file)
                st.audio(sound_file)
            
            # Get the data for the game of 2 teams
            df = playbyplayv2.PlayByPlayV2(games_id).get_data_frames()[0]

            #the following expression is specific to EventMsgType 1
            p = re.compile('(\s{2}|\' )([\w+ ]*)')

            # Get the PlayByPlay data from the Pacers game_id
            plays = playbyplayv2.PlayByPlayV2(games_id).get_normalized_dict()['PlayByPlay']

            #declare a few variables
            description = ''
            event_msg_action_types = {}
            event_msg_action_types1 = {}
            
            #loop over the play by play data
            for play in plays:
                # Report successful stats for the home team
                if play['EVENTMSGTYPE'] == 1:
                    description = play['HOMEDESCRIPTION'] if play['HOMEDESCRIPTION'] is not None else play['VISITORDESCRIPTION']
                    if description is not None:
                        try:
                            #do a bit of searching(regex) and a little character magic: underscores and upper case
                            event_msg_action = re.sub(' ', '_', p.search(description).groups()[1].rstrip()).upper()
                        except AttributeError:
                            event_msg_action = 'UNKNOWN'
                        #Add it to our dictionary
                        event_msg_action_types[event_msg_action] = play['EVENTMSGACTIONTYPE']
                
                # Report failed stats for the home team
                if play['EVENTMSGTYPE'] == 2:
                    match = []
                    if play['HOMEDESCRIPTION'] is not None: 
                        match = p.findall(play['HOMEDESCRIPTION'])
                        
                    if not match and play['VISITORDESCRIPTION'] is not None:
                        match = p.findall(play['VISITORDESCRIPTION'])
                        if len(match) & (play['HOMEDESCRIPTION'] is not None):
                            block = play['HOMEDESCRIPTION']
                    
                    if match:
                        event_msg_action = re.sub(' ', '_', match[0][1]).upper()
                        event_msg_action_types1[event_msg_action] = play['EVENTMSGACTIONTYPE']
                    
            event_msg_action_types1 = sorted(event_msg_action_types1.items(), key=operator.itemgetter(0))
            
            #sort it all
            event_msg_action_types = sorted(event_msg_action_types.items(), key=operator.itemgetter(0))
            
            # Spilt the headers into 2 sides
            left1, right1 = st.columns(2)
            left, right = st.columns(2)
            
            with left1:
                st.header("Success Field Goal")
            
            with right1:
                st.header("Missed Field Goal")
                
            # Output a class that we could plug into our code base: succesful home team stats
            for action in event_msg_action_types:
                with left:
                    st.write(f'\t{action[0]} = {action[1]}')
            
            # Output a class that we could plug into our code base: failed home team stats
            for action in event_msg_action_types1:
                with right:
                    st.write(f'\t{action[0]} = {action[1]}')
            
            # Add the Block stats to the right side
            with right:
                st.write(block)
            
            # Seperate the report and machine learning section
            st.write("---")
            
            # Random Forest NBA Playoff Prediction
            st.markdown("<h1 style='text-align: left; color: red;'>Machine Learning</h1>", unsafe_allow_html=True)
            
            #Random Forest Element
            overall_team_scoring = 0
            win_history = 0
            better_rank = 0
            regular_season_win = 0
            regular_season_win_percentage = 0
            road_play = 0
            home_play = 0

            # Collect home team player stats and sorted it to most 15 scored players contributed to the game
            team_player_stats = leagueplayerondetails.LeaguePlayerOnDetails(team_id=search_team_info(name=team_name)[0][0])
            team_player_stats = team_player_stats.get_data_frames()[0]
            team_player_stats = sqldf("select * from team_player_stats order by PTS DESC limit 15")
            
            # Collect opposite team player stats and sorted it to most 15 scored players contributed to the game
            opposing_team_player_stats = leagueplayerondetails.LeaguePlayerOnDetails(team_id=search_team_info(name=opposite_team_name)[0][0])
            opposing_team_player_stats = opposing_team_player_stats.get_data_frames()[0]
            opposing_team_player_stats = sqldf("select * from opposing_team_player_stats order by PTS DESC limit 15")
            
            # Print out their results
            st.subheader("Team Members Performance:")
            st.write(team_player_stats)
            
            st.subheader("Opposite Team Members Performance:")
            st.write(opposing_team_player_stats)
            
            # Print the points difference in dataframe
            comparing_player_stats = sqldf("WITH t1_row_numbers AS (SELECT ROW_NUMBER() OVER (ORDER BY TEAM_ID) as row_num, team_id, PTS FROM team_player_stats ), t2_row_numbers AS (SELECT ROW_NUMBER() OVER (ORDER BY TEAM_ID) as row_num, team_id, PTS FROM opposing_team_player_stats) SELECT t1_row_numbers.row_num, t1_row_numbers.PTS - t2_row_numbers.PTS as PTS_difference, CASE WHEN t1_row_numbers.PTS - t2_row_numbers.PTS > 0 THEN 1 ELSE 0 END as status FROM t1_row_numbers INNER JOIN t2_row_numbers ON t1_row_numbers.row_num = t2_row_numbers.row_num")
            
            st.subheader("Point difference between 2 teams best scorer from top to bottom and limitted to the best 15")
            st.write(comparing_player_stats)
            
            # Determine whose team has better team player scores in general
            overall_stats_result = comparing_player_stats['status'].value_counts()[1] - comparing_player_stats['status'].value_counts()[0]
            if overall_stats_result > 0:
                overall_team_scoring = 1
            elif overall_stats_result == 0 :
                overall_team_scoring = 0
            else:
                overall_team_scoring = -1
            
            # Determine whose team has more wins in the entire history they faced each other
            overall_win_result = games['WL'].value_counts()['W'] - games['WL'].value_counts()['L']
            if overall_win_result > 0:
                win_history = 1
            elif overall_win_result == 0 :
                win_history = 0
            else:
                win_history = -1
            
            # Extract the latest season standings and put it in dataframe
            league_standing = leaguestandingsv3.LeagueStandingsV3(season_type = 'Regular Season', season_nullable = "2022-2023")
            league_standing = league_standing.get_data_frames()[0]
            
            # Extract the necessary information such as TeamName, PlayoffRank, WINS, WinPCT, Last10Road, Last10Home from the dataframe for 2 teams 
            team_league_standing = sqldf(f"select TeamName, PlayoffRank, WINS, WinPCT, Last10Road, Last10Home from league_standing where TeamID = {search_team_info(name=team_name)[0][0]}")
            opposing_team_league_standing = sqldf(f"select TeamName, PlayoffRank, WINS, WinPCT, Last10Road, Last10Home from league_standing where TeamID = {search_team_info(name=opposite_team_name)[0][0]}")
            
            # Print the result
            st.subheader("Team Regular Season Performance")
            st.write(team_league_standing)
            
            st.subheader("Opposite Team Regular Season Performance")
            st.write(opposing_team_league_standing)
            
            # Extract information from each column and put it in list data structure for home team
            team_rank = team_league_standing.loc[:,"PlayoffRank"].tolist()
            team_win = team_league_standing.loc[:,"WINS"].tolist()
            team_win_percentage = team_league_standing.loc[:,"WinPCT"].tolist()
            team_road = team_league_standing.loc[:,"Last10Road"].tolist()
            team_home = team_league_standing.loc[:,"Last10Home"].tolist()
            
            # Extract information from each column and put it in list data structure for opposite team
            opposing_team_rank = opposing_team_league_standing.loc[:,"PlayoffRank"].tolist()
            opposing_team_win = opposing_team_league_standing.loc[:,"WINS"].tolist()
            opposing_team_win_percentage = opposing_team_league_standing.loc[:,"WinPCT"].tolist()
            opposing_team_road = opposing_team_league_standing.loc[:,"Last10Road"].tolist()
            opposing_team_home = opposing_team_league_standing.loc[:,"Last10Home"].tolist()
            
            # Compare the ranks from regular season to see who's better
            if int(team_rank[0]) - int(opposing_team_rank[0]) > 0:
                better_rank = 1
            elif int(team_rank[0]) - int(opposing_team_rank[0]) < 0:
                better_rank = -1
            else:
                better_rank = 0
            
            # Compare the number of wins from regular season to see who's better
            if int(team_win[0]) - int(opposing_team_win[0]) > 0:
                regular_season_win = 1
            elif int(team_win[0]) - int(opposing_team_win[0]) < 0:
                regular_season_win = -1
            else:
                regular_season_win = 0
            
            # Compare the win percentage from regular season to see who's better
            if int(team_win_percentage[0]) - int(opposing_team_win_percentage[0]) > 0:
                regular_season_win_percentage = 1
            elif int(team_win[0]) - int(opposing_team_win[0]) < 0:
                regular_season_win_percentage = -1
            else:
                regular_season_win_percentage = 0
            
            # Compare the last 10 road matches result from regular season to see who's better
            if int(team_road[0][0]) - int(opposing_team_road[0][0]) > 0:
                road_play = 1
            elif int(team_road[0][0]) - int(opposing_team_road[0][0]) < 0:
                road_play = -1
            else:
                road_play = 0
            
            # Compare the ast 10 home matches result from regular season to see who's better
            if int(team_home[0][0]) - int(opposing_team_home[0][0]) > 0:
                home_play = 1
            elif int(team_home[0][0]) - int(opposing_team_home[0][0]) < 0:
                home_play = -1
            else:
                home_play = 0
            
            # Put all the chosen variable of decison tree into 1 total random forest result
            random_forest_result = overall_team_scoring + win_history + better_rank + regular_season_win + regular_season_win_percentage + road_play + home_play
            
            # Check the status of the random result and print out prediction
            st.subheader("Simulation result if they face in the Playoff")
            if random_forest_result > 0:
                st.write(f"{search_team_info(name=team_name)[1][0]} is going to defeat {search_team_info(name=opposite_team_name)[1][0]} in the Playoff")
            elif random_forest_result < 0:
                st.write(f"{search_team_info(name=opposite_team_name)[1][0]} is going to defeat {search_team_info(name=team_name)[1][0]} in the Playoff")
            else:
                st.write("I'm not sure who's going to win in the playoff")
            
#Streamlit Multipage Creation
page_names_to_funcs = {
    "Introduction": introduction,
    "NBA Player Career Analysis": main_page,
    "NBA Team Seasonal Analysis": page2,
    "NBA Podcast": page3,
}

selected_page = st.sidebar.selectbox("Select a page", page_names_to_funcs.keys())
page_names_to_funcs[selected_page]()
