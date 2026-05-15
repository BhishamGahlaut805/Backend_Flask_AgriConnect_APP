from flask import Blueprint, Flask, request, jsonify
import plotly.graph_objs as go
from plotly.subplots import make_subplots
import plotly.io as pio
from flask_cors import CORS

#making blueprint for weather info
weather_bp = Blueprint('weather', __name__)
CORS(weather_bp)

def get_color(value, thresholds, colors):
    for threshold, color in zip(thresholds, colors):
        if value <= threshold:
            return color
    return colors[-1]

@weather_bp.route('/generateGraph', methods=['POST'])
def generate_graph():
    data = request.get_json()

    if not isinstance(data, list):
        return {"error": "Invalid data format, expected a list of dictionaries."}, 400

    try:
        times = [item['time'] for item in data]
        temperatures = [item['temperature'] for item in data]
        cloud_cover = [item['cloudCover'] for item in data]
        rain_chances = [item['rainChance'] for item in data]
        wind_speeds = [item['windSpeed'] for item in data]
        humidity = [item['humidity'] for item in data]
        moisture = [item['soil_moisture_0_to_1cm'] for item in data]
        soil_temperature = [item['soil_temperature_18cm'] for item in data]
        sunshine = float(data[0]['sunshine_duration'])  # Single value

        times = times[::2]
        temperatures = temperatures[::2]
        cloud_cover = cloud_cover[::2]
        rain_chances = rain_chances[::2]
        wind_speeds = wind_speeds[::2]
        humidity = humidity[::2]
        moisture = moisture[::2]
        soil_temperature = soil_temperature[::2]

        fig = make_subplots(
            rows=4, cols=1,
            subplot_titles=(
                'Hourly Temperature Overview',
                'Hourly Cloud Cover Overview',
                'Hourly Rain Chance Overview',
                'Hourly Wind Speed Overview'
            ),
            vertical_spacing=0.15
        )

        for i in range(1, 5):
            fig['layout'][f'annotations[{i-1}]']['font'] = dict(color='darkorange')

        fig.add_trace(go.Bar(
            x=times,
            y=temperatures,
            name='Temperature',
            marker=dict(color=[get_color(temp, [20, 30, 40], ['skyblue', 'lightgreen', 'orange', 'red']) for temp in temperatures]),
            text=[f'{temp}°C' for temp in temperatures],
            hoverinfo='text',
            width=0.7
        ), row=1, col=1)

        fig.add_trace(go.Bar(
            x=times,
            y=cloud_cover,
            name='Cloud Cover',
            marker=dict(color=[get_color(cc, [30, 60], ['lightyellow', 'lightblue', 'gray']) for cc in cloud_cover]),
            text=[f'{cc}%' for cc in cloud_cover],
            hoverinfo='text',
            width=0.7
        ), row=2, col=1)

        fig.add_trace(go.Bar(
            x=times,
            y=rain_chances,
            name='Rain Chance',
            marker=dict(color=[get_color(rc, [30, 60], ['lightgreen', 'yellow', 'red']) for rc in rain_chances]),
            text=[f'{rc}%' for rc in rain_chances],
            hoverinfo='text',
            width=0.7
        ), row=3, col=1)

        fig.add_trace(go.Bar(
            x=times,
            y=wind_speeds,
            name='Wind Speed',
            marker=dict(color=[get_color(ws, [10, 20], ['lightblue', 'lightgreen', 'orange']) for ws in wind_speeds]),
            text=[f'{ws} km/h' for ws in wind_speeds],
            hoverinfo='text',
            width=0.7
        ), row=4, col=1)

        fig.update_layout(
            title={
                'text': 'Hourly Weather Data Overview',
                'y': 0.98,
                'x': 0.5,
                'xanchor': 'center',
                'yanchor': 'top',
                'font': {
                    'size': 24,
                    'color': 'navy'
                }
            },
            height=1265,
            barmode='group',
            hovermode='x unified',
            showlegend=False,
            plot_bgcolor='white',
            font=dict(size=14),
            margin=dict(t=100, b=50, l=50, r=50),
            template='plotly_white'
        )

        fig.update_yaxes(title_text="Temperature (°C)", row=1, col=1, title_font=dict(color='darkcyan'))
        fig.update_yaxes(title_text="Cloud Cover (%)", row=2, col=1, title_font=dict(color='darkcyan'))
        fig.update_yaxes(title_text="Rain Chance (%)", row=3, col=1, title_font=dict(color='darkcyan'))
        fig.update_yaxes(title_text="Wind Speed (km/h)", row=4, col=1, title_font=dict(color='darkcyan'))

        fig.update_xaxes(title_text="Hourly Time", row=1, col=1, title_font=dict(color='darkcyan'))
        fig.update_xaxes(title_text="Hourly Time", row=2, col=1, title_font=dict(color='darkcyan'))
        fig.update_xaxes(title_text="Hourly Time", row=3, col=1, title_font=dict(color='darkcyan'))
        fig.update_xaxes(title_text="Hourly Time", row=4, col=1, title_font=dict(color='darkcyan'))

        graph_html = pio.to_html(fig, full_html=False)

        fig2 = make_subplots(
            rows=3, cols=1,
            subplot_titles=(
                'Hourly Humidity Overview',
                'Hourly Soil Moisture Overview',
                'Hourly Soil Temperature Overview'
            ),
            vertical_spacing=0.15
        )
        for i in range(1, 4):
            fig2['layout'][f'annotations[{i-1}]']['font'] = dict(color='darkorange')

        fig2.add_trace(go.Bar(
            x=times,
            y=humidity,
            name='Humidity',
            marker=dict(color=[get_color(h, [30, 60], ['lightyellow', 'lightblue', 'blue']) for h in humidity]),
            text=[f'{h}%' for h in humidity],
            hoverinfo='text',
            width=0.7
        ), row=1, col=1)

        fig2.add_trace(go.Bar(
            x=times,
            y=moisture,
            name='Soil Moisture',
            marker=dict(color=[get_color(m, [30, 60], ['brown', 'brown', 'darkbrown']) for m in moisture]),
            text=[f'{m}%' for m in moisture],
            hoverinfo='text',
            width=0.7
        ), row=2, col=1)

        fig2.add_trace(go.Bar(
            x=times,
            y=soil_temperature,
            name='Soil Temperature',
            marker=dict(color=[get_color(st, [20, 30, 40], ['skyblue', 'lightgreen', 'orange', 'red']) for st in soil_temperature]),
            text=[f'{st}°C' for st in soil_temperature],
            hoverinfo='text',
            width=0.7
        ), row=3, col=1)

        fig2.update_yaxes(title_text="Humidity (%)", row=1, col=1, title_font=dict(color='darkcyan'))
        fig2.update_yaxes(title_text="Soil Moisture (%)", row=2, col=1, title_font=dict(color='darkcyan'))
        fig2.update_yaxes(title_text="Soil Temperature (°C)", row=3, col=1, title_font=dict(color='darkcyan'))

        fig2.update_xaxes(title_text="Hourly Time", row=1, col=1, title_font=dict(color='darkcyan'))
        fig2.update_xaxes(title_text="Hourly Time", row=2, col=1, title_font=dict(color='darkcyan'))
        fig2.update_xaxes(title_text="Hourly Time", row=3, col=1, title_font=dict(color='darkcyan'))

        fig2.update_layout(
            title={
                'y': 0.98,
                'x': 0.5,
                'xanchor': 'center',
                'yanchor': 'top',
                'font': {
                    'size': 24,
                    'color': 'navy'
                }
            },
            height=1100,
            barmode='group',
            hovermode='x unified',
            showlegend=False,
            plot_bgcolor='white',
            font=dict(size=14),
            margin=dict(t=100, b=50, l=50, r=50),
            template='plotly_white'
        )

        graph_html2 = pio.to_html(fig2, full_html=False)

        pie_fig = go.Figure(go.Pie(
            labels=['Sunshine Time in hours For Today', 'Other'],
            values=[sunshine, 24 - sunshine],
            hole=0.4,
            marker=dict(colors=['gold', 'lightgray']),
            textinfo='label+percent',
            hoverinfo='label+value'
        ))

        pie_fig.update_layout(
            title={
                'text': 'Weather Parameters Graphical Representation',
                'y': 0.98,
                'x': 0.5,
                'xanchor': 'center',
                'yanchor': 'top',
                'font': {
                    'size': 24,
                    'color': 'navy'
                }
            },
            height=300,
            showlegend=False,
            plot_bgcolor='white',
            font=dict(size=14),
            margin=dict(t=100, b=50, l=50, r=50),
            template='plotly_white'
        )

        pie_chart_html = pio.to_html(pie_fig, full_html=False)

        # #Future weather Graphs
        tempmax = [item['tempDailyMax'] for item in data[:16]]
        tempmin = [item['tempDailyMin'] for item in data[:16]]
        days = [item['date'] for item in data[:16]]
        windspeedF= [float(item['windspeedF']) for item in data[:16]]
        rainChanceF= [float(item['rainChanceF']) for item in data[:16]]
        sunshineF = [float(item['sunshineF']) for item in data[:16]]

        fig3 = make_subplots(
            rows=1, cols=1,
            subplot_titles=('Daily Temperature Overview',),
            vertical_spacing=0.15
        )
        fig3.add_trace(go.Bar(
            x=days,
            y=tempmax,
            name='Max Temperature',
            marker=dict(color=[get_color(temp, [20, 30, 40], ['lightgreen', 'brown', 'cyan', 'purple']) for temp in tempmax]),
            text=[f'{temp}°C' for temp in tempmax],
            hoverinfo='text',
            width=0.5
        ), row=1, col=1)
        fig3.add_trace(go.Bar(
            x=days,
            y=tempmin,
            name='Min Temperature',
            marker=dict(color=[get_color(temp, [10, 20, 30], ['yellow', 'skyblue', 'lightblue']) for temp in tempmin]),
            text=[f'{temp}°C' for temp in tempmin],
            hoverinfo='text',
            width=0.5
        ), row=1, col=1)
        fig3.update_layout(
            xaxis_title='Days',
            yaxis_title='Temperature (°C)',
            template='plotly_white',
            title={
                'text': 'Average Daily Max and Min Temperatures',
                'y': 0.98,
                'x': 0.5,
                'xanchor': 'center',
                'yanchor': 'top',
                'font': {
                    'size': 24,
                    'color': 'navy'
                }
            },
            barmode='group',
            hovermode='x unified',
            showlegend=True,
            plot_bgcolor='white',
            font=dict(size=14),
            margin=dict(t=100, b=50, l=50, r=50),
            width=1000  # Increase the width of the figure
        )
        fig3.update_yaxes(title_text="Temperature (°C)", row=1, col=1, title_font=dict(color='darkcyan'))
        fig3.update_xaxes(title_text="Days", row=1, col=1, title_font=dict(color='darkcyan'))

        graph_html3 = pio.to_html(fig3, full_html=False)

        #some more Future Graphs

        fig4 = make_subplots(
            rows=3, cols=1,
            subplot_titles=(
                'Daily Wind Speed Overview',
                'Daily Chance of Rain Overview',
                'Daily Sunlight Duration Overview'
            ),
            vertical_spacing=0.15
        )
        for i in range(1, 4):
            fig4['layout'][f'annotations[{i-1}]']['font'] = dict(color='darkorange')

        fig4.add_trace(go.Bar(
            x=days,
            y=windspeedF,
            name='Wind speed',
            marker=dict(color=[get_color(h, [30, 60], ['lightyellow', 'lightblue', 'blue']) for h in windspeedF]),
            text=[f'{h}Km/h' for h in windspeedF],
            hoverinfo='text',
            width=0.7
        ), row=1, col=1)

        fig4.add_trace(go.Bar(
            x=days,
            y=rainChanceF,
            name='Chance of Rain',
            marker=dict(color=[get_color(m, [30, 60], ['lightgreen', 'brown', 'skyblue']) for m in rainChanceF]),
            text=[f'{m}%' for m in rainChanceF],
            hoverinfo='text',
            width=0.7
        ), row=2, col=1)

        fig4.add_trace(go.Bar(
            x=days,
            y=sunshineF,
            name='Sunlight Duration in Hours',
            marker=dict(color=[get_color(st, [20, 30, 40], ['skyblue', 'lightgreen', 'orange', 'red']) for st in sunshineF]),
            text=[f'{st}Hours' for st in sunshineF],
            hoverinfo='text',
            width=0.7
        ), row=3, col=1)

        fig4.update_yaxes(title_text="Wind Speed(Km/h)", row=1, col=1, title_font=dict(color='darkcyan'))
        fig4.update_yaxes(title_text="Chance of Rain(%)", row=2, col=1, title_font=dict(color='darkcyan'))
        fig4.update_yaxes(title_text="Sunlight Duration(hours)", row=3, col=1, title_font=dict(color='darkcyan'))

        fig4.update_xaxes(title_text="Days", row=1, col=1, title_font=dict(color='darkcyan'))
        fig4.update_xaxes(title_text="Days", row=2, col=1, title_font=dict(color='darkcyan'))
        fig4.update_xaxes(title_text="Days", row=3, col=1, title_font=dict(color='darkcyan'))

        fig4.update_layout(
            title={
                'y': 0.98,
                'x': 0.5,
                'xanchor': 'center',
                'yanchor': 'top',
                'font': {
                    'size': 24,
                    'color': 'navy'
                }
            },
            height=1100,
            barmode='group',
            hovermode='x unified',
            showlegend=False,
            plot_bgcolor='white',
            font=dict(size=14),
            margin=dict(t=100, b=50, l=50, r=50),
            template='plotly_white'
        )

        graph_html4 = pio.to_html(fig4, full_html=False)

        return jsonify({"graph_html": graph_html,
                        "graph_html2": graph_html2,
                        "pie_chart_html": pie_chart_html,
                        'graph_html3':graph_html3,
                       'graph_html4':graph_html4}), 200

    except KeyError as e:
        return {"error": f"Missing key in data: {e}"}, 400


@weather_bp.route('/generateSeasonal', methods=['POST'])
def generate():
    data = request.json
    date = data.get('date')
    temp_max = data.get('tempMax')
    temp_min = data.get('tempMin')
    wind_speed = data.get('windSpeed')
    precipitation = data.get('rain')

    # Create the main weather plot
    fig = go.Figure()

    # Add traces for weather data
    fig.add_trace(go.Scatter(
        x=date, y=temp_max,
        mode='lines+markers',
        name='Max Temp (°C)',
        line=dict(color='firebrick', width=2),
        marker=dict(size=6),
        hoverinfo='x+y'
    ))

    fig.add_trace(go.Scatter(
        x=date, y=temp_min,
        mode='lines+markers',
        name='Min Temp (°C)',
        line=dict(color='royalblue', width=2),
        marker=dict(size=6),
        hoverinfo='x+y'
    ))

    fig.add_trace(go.Scatter(
        x=date, y=wind_speed,
        mode='lines+markers',
        name='Wind Speed (m/s)',
        line=dict(color='green', width=2),
        marker=dict(size=6),
        hoverinfo='x+y'
    ))

    fig.add_trace(go.Scatter(
        x=date, y=precipitation,
        mode='lines+markers',
        name='Precipitation (mm)',
        line=dict(color='purple', width=2),
        marker=dict(size=6),
        hoverinfo='x+y'
    ))

    # Update layout for the weather plot
    fig.update_layout(
        title='Seasonal Weather Data',
        title_font=dict(size=24, color='darkblue'),
        xaxis_title='Date',
        yaxis_title='Values',
        xaxis=dict(showgrid=True),
        yaxis=dict(showgrid=True),
        legend=dict(title='Parameters', orientation='h'),
        margin=dict(l=40, r=40, t=40, b=40),
        plot_bgcolor='rgba(245, 245, 245, 0.8)',
        hovermode='closest',
        height=500  # Increase the height of the layout
    )


    # Convert both figures to HTML
    weather_graph_html = pio.to_html(fig, full_html=False)

    return jsonify({"graph_html": weather_graph_html,}), 200


