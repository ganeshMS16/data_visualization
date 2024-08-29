from langchain_community.llms import OpenAI
from langchain_experimental.agents import create_pandas_dataframe_agent
import pandas as pd
from dotenv import load_dotenv
import json
import streamlit as st
import os
import plotly.express as px

# Load environment variables
load_dotenv()

# Set API key for OpenAI
openai_api_key = os.getenv("OPENAI_API_KEY")
print(f"API Key: {openai_api_key}")

def csv_tool(filename: str):
    df = pd.read_csv(filename)
    return create_pandas_dataframe_agent(OpenAI(temperature=0, openai_api_key=openai_api_key), df, verbose=True, allow_dangerous_code=True)

def ask_agent(agent, query):
    prompt = (
        """
        Let's decode the way to respond to the queries. The responses depend on the type of information requested in the query.

        1. If the query requires a table, format your answer like this:
           {"table": {"columns": ["column1", "column2", ...], "data": [["value1", "value2", ...], ["value1", "value2", ...], ...]}}

        2. For a bar chart, respond like this:
           {"bar": {"columns": ["A", "B", "C", ...], "data": [["25", "24", "10", ...], ["12", "18", "7", ...], ...]}}

        3. For a line chart, respond like this:
           {"line": {"columns": ["A", "B", "C", ...], "data": [["25", "24", "10", ...], ["12", "18", "7", ...], ...]}}

        4. For a scatter plot, respond like this:
           {"scatter": {"columns": ["X", "Y"], "data": [["1", "2"], ["3", "4"], ...]}}

        5. For a pie chart, respond like this:
           {"pie": {"labels": ["A", "B", "C", ...], "data": ["25", "24", "10", ...]}}

        6. For a treemap chart, respond like this:
           {"treemap": {"columns": ["Category", "Value"], "data": [["A", "10"], ["B", "20"], ...]}}

        7. For a Pareto chart, respond like this:
           {"pareto": {"columns": ["Category", "Value"], "data": [["A", "30"], ["B", "70"], ...]}}

        8. For a heatmap chart, respond like this:
           {"heatmap": {"columns": ["X", "Y", "Value"], "data": [["1", "2", "3"], ["4", "5", "6"], ...]}}

        9. For a geo chart, respond like this:
           {"geo": {"columns": ["Latitude", "Longitude", "Size"], "data": [["37.7749", "-122.4194", "20"], ...]}}

        10. For a waterfall chart, respond like this:
            {"waterfall": {"columns": ["Category", "Value"], "data": [["Start", "100"], ["Increase", "50"], ["Decrease", "-20"], ...]}}

        11. For a donut chart, respond like this:
            {"donut": {"labels": ["A", "B", "C", ...], "data": ["25", "24", "10", ...]}}

        12. For a funnel chart, respond like this:
            {"funnel": {"columns": ["Stage", "Value"], "data": [["Initial", "100"], ["Middle", "50"], ["End", "10"], ...]}}

        13. For a bubble chart, respond like this:
            {"bubble": {"columns": ["X", "Y", "Size", "Color"], "data": [["1", "2", "10", "Red"], ["3", "4", "20", "Blue"], ...]}}

        14. For a histogram chart, respond like this:
            {"histogram": {"columns": ["X"], "data": [["1"], ["2"], ["3"], ...]}}

        15. For a candlestick chart, respond like this:
            {"candlestick": {"columns": ["Date", "Open", "High", "Low", "Close"], "data": [["2024-08-01", "100", "110", "90", "105"], ...]}}

        16. For an area chart, respond like this:
            {"area": {"columns": ["X", "Y"], "data": [["1", "10"], ["2", "15"], ["3", "20"], ...]}}

        17. For a KPI, respond like this:
            {"kpi": {"label": "Metric", "value": "100", "delta": "+10"}}

        18. For a radar chart, respond like this:
            {"radar": {"columns": ["Metric", "Value"], "data": [["Speed", "70"], ["Quality", "80"], ["Cost", "90"], ...]}}

        19. For a Sankey chart, respond like this:
            {"sankey": {"nodes": ["Node1", "Node2", "Node3", ...], "source": ["0", "1", "2", ...], "target": ["1", "2", "3", ...], "value": ["5", "10", "15", ...]}}

        20. For a box plot, respond like this:
            {"box": {"columns": ["Category", "Value"], "data": [["A", "10"], ["A", "15"], ["B", "20"], ["B", "25"], ...]}}

        20. For a plain question that doesn't need a chart or table, your response should be:
           {"answer": "Your answer goes here"}

        21. If the answer is not known or available, respond with:
           {"answer": "I do not know."}

        Return all output as a string. Remember to encase all strings in the "columns" list and data list in double quotes.
        """
        + query
    )
    return agent.run(prompt)

    response = agent.run(prompt)
    return str(response)

def decode_response(response: str) -> dict:
    return json.loads(response)

def plot_with_plotly(response_dict: dict):
    """This function handles various chart types using Plotly based on the response from the model."""

    # Bar Chart
    if "bar" in response_dict:
        data = response_dict["bar"]
        df = pd.DataFrame({
            col: [x[i] if isinstance(x, list) else x for x in data['data']]
            for i, col in enumerate(data['columns'])
        })
        fig = px.bar(df, x=df.columns[0], y=df.columns[1])
        st.plotly_chart(fig)

    # Line Chart
    elif "line" in response_dict:
        data = response_dict["line"]
        df = pd.DataFrame({
            col: [x[i] if isinstance(x, list) else x for x in data['data']]
            for i, col in enumerate(data['columns'])
        })
        fig = px.line(df, x=df.columns[0], y=df.columns[1])
        st.plotly_chart(fig)

    # Scatter Plot
    elif "scatter" in response_dict:
        data = response_dict["scatter"]
        df = pd.DataFrame(data["data"], columns=data["columns"])
        fig = px.scatter(df, x=df.columns[0], y=df.columns[1])
        st.plotly_chart(fig)

    # Pie Chart
    elif "pie" in response_dict:
        data = response_dict["pie"]
        df = pd.DataFrame({
            "labels": data["labels"],
            "values": data["data"]
        })
        fig = px.pie(df, values='values', names='labels')
        st.plotly_chart(fig)

    # Treemap Chart
    elif "treemap" in response_dict:
        data = response_dict["treemap"]
        df = pd.DataFrame(data["data"], columns=data["columns"])
        fig = px.treemap(df, path=[data["columns"][0]], values=data["columns"][1])
        st.plotly_chart(fig)

    # Pareto Chart
    elif "pareto" in response_dict:
        data = response_dict["pareto"]
        df = pd.DataFrame(data["data"], columns=data["columns"])
        df['cum_percentage'] = df[data["columns"][1]].cumsum() / df[data["columns"][1]].sum() * 100
        fig = px.bar(df, x=data["columns"][0], y=data["columns"][1])
        fig.add_scatter(x=df[data["columns"][0]], y=df['cum_percentage'], mode='lines', name='Cumulative Percentage')
        st.plotly_chart(fig)

    # Heatmap Chart
    elif "heatmap" in response_dict:
        data = response_dict["heatmap"]
        df = pd.DataFrame(data["data"], columns=data["columns"])
        fig = px.imshow(df, labels=dict(x=data["columns"][0], y=data["columns"][1], color="Value"))
        st.plotly_chart(fig)

    # Geo Chart
    elif "geo" in response_dict:
        data = response_dict["geo"]
        df = pd.DataFrame(data["data"], columns=data["columns"])
        fig = px.scatter_geo(df, lat=df.columns[0], lon=df.columns[1], size=data["columns"][2])
        st.plotly_chart(fig)

    # Waterfall Chart
    elif "waterfall" in response_dict:
        data = response_dict["waterfall"]
        df = pd.DataFrame(data["data"], columns=data["columns"])
        fig = px.waterfall(df, x=df.columns[0], y=df.columns[1])
        st.plotly_chart(fig)

    # Donut Chart
    elif "donut" in response_dict:
        data = response_dict["donut"]
        df = pd.DataFrame({
            "labels": data["labels"],
            "values": data["data"]
        })
        fig = px.pie(df, values='values', names='labels', hole=0.4)
        st.plotly_chart(fig)

    # Funnel Chart
    elif "funnel" in response_dict:
        data = response_dict["funnel"]
        df = pd.DataFrame(data["data"], columns=data["columns"])
        fig = px.funnel(df, x=df.columns[0], y=df.columns[1])
        st.plotly_chart(fig)

    # Bubble Chart
    elif "bubble" in response_dict:
        data = response_dict["bubble"]
        df = pd.DataFrame(data["data"], columns=data["columns"])
        fig = px.scatter(df, x=df.columns[0], y=df.columns[1], size=data["columns"][2], color=data["columns"][3])
        st.plotly_chart(fig)

    # Histogram Chart
    elif "histogram" in response_dict:
        data = response_dict["histogram"]
        df = pd.DataFrame(data["data"], columns=data["columns"])
        fig = px.histogram(df, x=df.columns[0])
        st.plotly_chart(fig)

    # Candlestick Chart
    elif "candlestick" in response_dict:
        data = response_dict["candlestick"]
        df = pd.DataFrame(data["data"], columns=data["columns"])
        fig = px.candlestick(df, x=df.columns[0], open=df.columns[1], high=df.columns[2], low=df.columns[3], close=df.columns[4])
        st.plotly_chart(fig)

    # Area Chart
    elif "area" in response_dict:
        data = response_dict["area"]
        df = pd.DataFrame(data["data"], columns=data["columns"])
        fig = px.area(df, x=df.columns[0], y=df.columns[1])
        st.plotly_chart(fig)

    # KPI Chart
    elif "kpi" in response_dict:
        data = response_dict["kpi"]

        # Check if the value is a multiline string and parse it if necessary
        if isinstance(data["value"], str) and "\n" in data["value"]:
            # Split the string into lines and handle them separately
            lines = data["value"].split('\n')
            for line in lines:
                if line.strip():  # Ensure the line is not empty
                    parts = line.split()
                    if len(parts) >= 2:
                        label = f"{data['label']} - {parts[0]}"
                        value = parts[1]
                        st.metric(label=label, value=value, delta=data.get("delta"))
        else:
            # Display a single KPI
            st.metric(label=data["label"], value=data["value"], delta=data.get("delta"))

    # Radar Chart
    elif "radar" in response_dict:
        data = response_dict["radar"]
        df = pd.DataFrame(data["data"], columns=data["columns"])
        fig = px.line_polar(df, r=df[data["columns"][1]], theta=df[data["columns"][0]], line_close=True)
        st.plotly_chart(fig)

    # Sankey Chart
    elif "sankey" in response_dict:
        data = response_dict["sankey"]
        fig = px.sankey(node=dict(label=data['nodes']), link=dict(source=data['source'], target=data['target'], value=data['value']))
        st.plotly_chart(fig)

     # Box Plot
    elif "boxplot" in response_dict:
        data = response_dict["boxplot"]
        df = pd.DataFrame(data["data"], columns=data["columns"])
        fig = px.box(df, y=df.columns[0], points="all", title=f'Box Plot of {data["by"]}')
        st.plotly_chart(fig)


def write_answer(response_dict: dict):
    """This function writes the response or renders charts based on the response from the agent."""
    if "answer" in response_dict:
        st.write(response_dict["answer"])

    elif "table" in response_dict:
        data = response_dict["table"]
        df = pd.DataFrame(data["data"], columns=data["columns"])
        st.table(df)
    
    else:
        plot_with_plotly(response_dict)

st.set_page_config(page_title="👨‍💻 Talk with your CSV")
st.title("👨‍💻 Talk with your CSV")

st.write("Please upload your CSV file below.")
data = st.file_uploader("Upload a CSV", type="csv")

query = st.text_area("Send a Message")

if st.button("Submit Query", type="primary"):
    if data is not None and query.strip():
        agent = csv_tool(data)
        response = ask_agent(agent=agent, query=query)
        decoded_response = decode_response(response)
        write_answer(decoded_response)
    else:
        st.write("Please upload a CSV file and enter a query.")
