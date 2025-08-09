import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import json
import os
from datetime import datetime, timedelta
from sklearn.linear_model import LinearRegression
from fpdf import FPDF
import base64
import hashlib
from twilio.rest import Client
import streamlit as st





  # or a hard-coded list

# and remove or comment out the sidebar config input section


# Constants and Setup ------------------------------------------------------
MACHINES = ['3D Printer', 'CNC Mill', 'Air Compressor']
SENSORS = ['temperature', 'vibration', 'current']
STATUS_COLORS = {'Normal': 'green', 'Warning': 'orange', 'Critical': 'red', 'Unknown': 'grey'}
DATA_DIR = "data_storage"

DEFAULT_THRESHOLDS = {
    '3D Printer': {'temperature': (70.0, 90.0), 'vibration': (10.0, 20.0), 'current': (5.0, 8.0)},
    'CNC Mill': {'temperature': (80.0, 100.0), 'vibration': (15.0, 30.0), 'current': (10.0, 15.0)},
    'Air Compressor': {'temperature': (80.0, 95.0), 'vibration': (10.0, 18.0), 'current': (7.0, 13.0)},
}

WEIGHTS = {'temperature': 0.4, 'vibration': 0.3, 'current': 0.3}  # For combined health score

# Authentication Setup -----------------------------------------------------
USERS = {
    "admin": {
        "password_hash": hashlib.sha256("adminpass".encode()).hexdigest(),
        "role": "admin"
    },
    "staff": {
        "password_hash": hashlib.sha256("staffpass".encode()).hexdigest(),
        "role": "staff"
    }
}

def verify_password(username, password):
    if username in USERS:
        hashed = hashlib.sha256(password.encode()).hexdigest()
        return hashed == USERS[username]['password_hash']
    return False

def login():
    st.sidebar.title("🔐 Login")
    username = st.sidebar.text_input("Username")
    password = st.sidebar.text_input("Password", type="password")
    if st.sidebar.button("Login"):
        if verify_password(username, password):
            st.session_state['logged_in'] = True
            st.session_state['user'] = username
            st.session_state['role'] = USERS[username]['role']
            st.sidebar.success(f"Logged in as {username} ({st.session_state['role']})")
        else:
            st.sidebar.error("Invalid username or password")
    if 'logged_in' not in st.session_state or not st.session_state['logged_in']:
        st.stop()

# Persistent Storage Helpers -------------------------------------------------
def ensure_data_dir():
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)

def save_operator_notes(notes_dict):
    ensure_data_dir()
    path = os.path.join(DATA_DIR, 'operator_notes.json')
    with open(path, 'w') as f:
        json.dump(notes_dict, f)

def load_operator_notes():
    ensure_data_dir()
    path = os.path.join(DATA_DIR, 'operator_notes.json')
    if os.path.exists(path):
        with open(path, 'r') as f:
            return json.load(f)
    return {m: "" for m in MACHINES}

def save_sensor_data(df: pd.DataFrame):
    ensure_data_dir()
    path = os.path.join(DATA_DIR, 'sensor_data.csv')
    if os.path.exists(path):
        df_existing = pd.read_csv(path)
        df = pd.concat([df_existing, df], ignore_index=True)
        df.drop_duplicates(subset=['timestamp','machine','sensor'], keep='last', inplace=True)
    df.to_csv(path, index=False)

def load_sensor_data():
    ensure_data_dir()
    path = os.path.join(DATA_DIR, 'sensor_data.csv')
    if os.path.exists(path):
        df = pd.read_csv(path)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        return df
    return pd.DataFrame(columns=['timestamp','machine','sensor','value'])

def save_alert_log(alert_msg):
    ensure_data_dir()
    path = os.path.join(DATA_DIR, 'alert_log.txt')
    with open(path, 'a') as f:
        f.write(f"{datetime.now().isoformat()} - {alert_msg}\n")

def load_alert_log():
    ensure_data_dir()
    path = os.path.join(DATA_DIR, 'alert_log.txt')
    if os.path.exists(path):
        with open(path, 'r') as f:
            return f.read()
    return ""

def reset_alert_state():
    st.session_state['last_status'] = {}

# Twilio SMS send helper ----------------------------------------------------
def send_sms_alert(to_number, body, twilio_sid, twilio_token, twilio_number):
    client = Client(twilio_sid, twilio_token)
    message = client.messages.create(
        body=body,
        from_=twilio_number,
        to=to_number
    )
    return message.sid

# Rule Engine ----------------------------------------------------------------
def classify_status(val, sensor, machine, thresholds):
    if val is None or pd.isna(val):
        return 'Unknown'
    if machine not in thresholds or sensor not in thresholds[machine]:
        alert = f"DEBUG: No rule defined for {machine} sensor {sensor}, defaulting to Normal"
        print(alert)
        save_alert_log(alert)
        return 'Normal'
    warn, crit = thresholds[machine][sensor]
    try:
        val = float(val)
    except:
        return 'Unknown'
    if val >= crit:
        return 'Critical'
    elif val >= warn:
        return 'Warning'
    else:
        return 'Normal'

def is_stale(latest_timestamp, threshold_seconds=40):
    if latest_timestamp is None:
        return True
    return (datetime.now() - latest_timestamp).total_seconds() > threshold_seconds

def plot_history(data, machine, sensor):
    df = data[(data['machine'] == machine) & (data['sensor'] == sensor)]
    if df.empty:
        st.write("No data available.")
        return
    fig = px.line(df, x='timestamp', y='value',
                  markers=True,
                  title=f"{machine} - {sensor.capitalize()} (Last 60 minutes)")
    st.plotly_chart(fig, use_container_width=True)

def plot_combined_health(df, machine, thresholds):
    df_filtered = df[(df['machine']==machine) & (df['timestamp'] >= datetime.now() - timedelta(minutes=60))]
    if df_filtered.empty:
        st.write("No sensor data for Combined Health Score")
        return

    latest_vals = df_filtered.groupby('sensor').apply(lambda g: g.sort_values('timestamp').iloc[-1]['value'] if not g.empty else np.nan)
    total_weight = 0
    total_score = 0
    for sensor in SENSORS:
        val = latest_vals.get(sensor, np.nan)
        warn, crit = thresholds[machine][sensor]
        if pd.isna(val):
            continue
        val_norm = min(max(val, 0), crit) / crit
        score = (1 - val_norm) * 100
        w = WEIGHTS[sensor]
        total_score += score * w
        total_weight += w
    if total_weight == 0:
        st.write("Combined Health Score: N/A (no sensor data)")
        return

    combined_score = total_score / total_weight
    if combined_score > 80:
        color = 'red'
    elif combined_score > 60:
        color = 'orange'
    elif combined_score > 40:
        color = 'yellow'
    else:
        color = 'green'
    st.markdown(f"<h3 style='color:{color}'>Combined Health Score: {combined_score:.1f}/100</h3>", unsafe_allow_html=True)

def estimate_time_to_critical(values, timestamps, warn, crit):
    values = np.array(values)
    timestamps = pd.to_datetime(timestamps)
    if len(values) < 2 or np.any(pd.isna(values)):
        return None
    x = (timestamps - timestamps.min()).dt.total_seconds().values.reshape(-1, 1)
    y = values.reshape(-1,1)
    lr = LinearRegression()
    lr.fit(x, y)
    slope = lr.coef_[0][0]
    intercept = lr.intercept_[0]

    if slope <= 0:
        return None

    est_seconds = (crit - intercept) / slope
    if est_seconds < 0:
        return None

    est_time = timestamps.min() + timedelta(seconds=est_seconds)
    time_remaining = est_time - timestamps.max()
    if time_remaining.total_seconds() < 0:
        return None

    return time_remaining

def moving_average_slope(values, window=5):
    if len(values) < window + 1:
        return None
    ma = pd.Series(values).rolling(window).mean().dropna()
    if len(ma) < 2:
        return None
    return ma.iloc[-1] - ma.iloc[-2]

def export_pdf(machine, df_data, alert_log, notes):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", 'B', 16)
    pdf.cell(0, 10, f'Machine Health Report: {machine}', ln=1, align='C')
    pdf.set_font("Arial", '', 12)
    pdf.cell(0, 10, f'Report Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}', ln=1)
    pdf.ln(5)
    pdf.set_font("Arial", 'B', 14)
    pdf.cell(0, 10, "Latest Sensor Readings (Last 60 Minutes):", ln=1)
    pdf.set_font("Arial", '', 12)
    for sensor in SENSORS:
        sub_df = df_data[(df_data['machine']==machine) & (df_data['sensor']==sensor)]
        if sub_df.empty:
            pdf.cell(0, 10, f"{sensor.capitalize()}: No data", ln=1)
        else:
            latest_row = sub_df.sort_values('timestamp').iloc[-1]
            val = latest_row['value']
            ts = latest_row['timestamp']
            pdf.cell(0, 10, f"{sensor.capitalize()}: {val} (at {ts})", ln=1)
    pdf.ln(5)
    pdf.set_font("Arial", 'B', 14)
    pdf.cell(0, 10, "Alert Log:", ln=1)
    pdf.set_font("Arial", '', 12)
    if alert_log.strip() == '':
        pdf.cell(0, 10, "No alerts.", ln=1)
    else:
        alerts = [line for line in alert_log.splitlines() if machine in line]
        if alerts:
            for alert in alerts[-10:]:
                pdf.cell(0, 10, alert, ln=1)
        else:
            pdf.cell(0, 10, "No alerts for this machine.", ln=1)
    pdf.ln(5)
    pdf.set_font("Arial", 'B', 14)
    pdf.cell(0, 10, "Operator Notes:", ln=1)
    pdf.set_font("Arial", '', 12)
    note = notes.get(machine, "")
    if note.strip() == "":
        pdf.cell(0, 10, "No notes recorded.", ln=1)
    else:
        pdf.multi_cell(0, 10, note)
    return pdf.output(dest='S').encode('latin1')

def get_pdf_download_link(pdf_bytes, machine):
    b64 = base64.b64encode(pdf_bytes).decode()
    href = f'<a href="data:application/octet-stream;base64,{b64}" download="report_{machine}.pdf">Download PDF Report for {machine}</a>'
    return href

def validate_uploaded_df(df):
    expected_cols = {'timestamp', 'machine', 'sensor', 'value'}
    cols = set(df.columns)
    if not expected_cols.issubset(cols):
        raise ValueError(f"Missing required columns. Found columns: {cols}. Expected: {expected_cols}")
    unknown_machines = set(df['machine'].unique()) - set(MACHINES)
    unknown_sensors = set(df['sensor'].unique()) - set(SENSORS)
    if unknown_machines:
        raise ValueError(f"Unknown machines found in uploaded data: {unknown_machines}")
    if unknown_sensors:
        raise ValueError(f"Unknown sensors found in uploaded data: {unknown_sensors}")
    df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
    if df['timestamp'].isnull().any():
        raise ValueError("Some timestamps could not be parsed.")
    df['value'] = pd.to_numeric(df['value'], errors='coerce')
    if df['value'].isnull().any():
        raise ValueError("Some sensor values are not numeric or missing.")
    df.sort_values('timestamp', inplace=True)
    df.drop_duplicates(subset=['timestamp','machine','sensor'], keep='last', inplace=True)
    return df

def mqtt_simulation_stub():
    st.info("MQTT simulation is a stretch goal. To implement, connect to a broker publishing sensor data every 30s.\nCurrently no active MQTT connection.")

def llm_explain_trend(machine, sensor, recent_df):
    if recent_df.empty:
        return "No data available for explanation."
    latest_val = recent_df['value'].iloc[-1]
    status = classify_status(latest_val, sensor, machine, st.session_state['thresholds'])
    explanation = f"Latest reading for {sensor} is {latest_val:.2f}, which puts the sensor status at '{status}'."
    if status == 'Critical':
        explanation += " This indicates a critical condition that requires immediate attention."
    elif status == 'Warning':
        explanation += " This is a warning; monitor closely."
    else:
        explanation += " The system is functioning normally."
    return explanation

def main():
    st.set_page_config(page_title="Workshop-Watch Dashboard", layout="wide")

    if 'logged_in' not in st.session_state or not st.session_state['logged_in']:
        login()
        return

    if 'thresholds' not in st.session_state:
        st.session_state['thresholds'] = DEFAULT_THRESHOLDS.copy()
    if 'sensor_data' not in st.session_state:
        st.session_state['sensor_data'] = load_sensor_data()
    if 'operator_notes' not in st.session_state:
        st.session_state['operator_notes'] = load_operator_notes()
    if 'last_status' not in st.session_state:
        reset_alert_state()

    role = st.session_state.get('role', 'staff')

    st.title("Workshop-Watch: Industrial IoT Monitoring Dashboard")
    st.sidebar.title("Controls")

    st.sidebar.markdown(f"**User:** {st.session_state.get('user')} ({role})")
    if st.sidebar.button("Logout"):
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        st.rerun()

    # Twilio SMS Alert Config

    # Upload sensor data
    st.sidebar.header("Upload Sensor Data")
    uploaded_file = st.sidebar.file_uploader("Upload JSON, CSV or Excel (xlsx)", type=['json','csv','xlsx','xls'])
    if uploaded_file is not None:
        try:
            if uploaded_file.name.endswith('.json'):
                df_uploaded = pd.read_json(uploaded_file)
            elif uploaded_file.name.endswith('.csv'):
                df_uploaded = pd.read_csv(uploaded_file)
            else:
                df_uploaded = pd.read_excel(uploaded_file)
            df_uploaded = validate_uploaded_df(df_uploaded)
            st.session_state['sensor_data'] = pd.concat([st.session_state['sensor_data'], df_uploaded], ignore_index=True)
            st.session_state['sensor_data'].drop_duplicates(subset=['timestamp','machine','sensor'], keep='last', inplace=True)
            save_sensor_data(st.session_state['sensor_data'])
            st.sidebar.success("File uploaded and data ingested successfully!")
        except Exception as e:
            st.sidebar.error(f"Error processing file: {e}")

    # Simulate data button
    if st.sidebar.button("Simulate new data (30s interval)"):
        now = datetime.now()
        new_entries = []
        for m in MACHINES:
            for s in SENSORS:
                base_val = np.mean(DEFAULT_THRESHOLDS[m][s]) * 0.75
                noise = np.random.normal(0, 5)
                val = max(0, base_val + noise)
                new_entries.append({'timestamp': now, 'machine': m, 'sensor': s, 'value': val})
        new_df = pd.DataFrame(new_entries)
        st.session_state['sensor_data'] = pd.concat([st.session_state['sensor_data'], new_df], ignore_index=True)
        save_sensor_data(st.session_state['sensor_data'])
        st.success("Simulated new data added!")

    if role == 'admin':
        st.sidebar.header("Edit Thresholds (Admin)")
        for m in MACHINES:
            st.sidebar.subheader(m)
            for s in SENSORS:
                warn_val, crit_val = st.sidebar.slider(
                    f"{m} - {s} thresholds (Warning, Critical)",
                    min_value=0.0,
                    max_value=200.0,
                    value=st.session_state['thresholds'][m][s],
                    step=0.1,
                    key=f"thresh_{m}_{s}"
                )
                if warn_val >= crit_val:
                    st.sidebar.error("Warning threshold must be less than Critical threshold.")
                else:
                    st.session_state['thresholds'][m][s] = (warn_val, crit_val)

    # Alert simulator with SMS integration
    def simulate_alert(status, machine, sensor, value):
        key = f"{machine}_{sensor}"
        last_status = st.session_state.get('last_status', {})
        previous = last_status.get(key)

        if status == 'Critical' and previous != 'Critical':
            alert_msg = f"CRITICAL ALERT! {machine} - {sensor}: {value}"
            st.error(alert_msg)
            print(alert_msg)
            save_alert_log(alert_msg)

            # Send SMS via Twilio if config is valid
            if all([twilio_sid, twilio_token, twilio_number]) and sms_recipients:
                body = f"Critical alert on machine '{machine}', sensor '{sensor}': value = {value}"
                for number in sms_recipients:
                    try:
                        sid = send_sms_alert(number, body, twilio_sid, twilio_token, twilio_number)
                        st.sidebar.success(f"SMS alert sent Successfully")
                    except Exception as e:
                        st.sidebar.error(f"Failed to send SMS ")
            else:
                st.sidebar.warning("SMS alert NOT sent: incomplete Twilio configuration or phone number(s).")

        last_status[key] = status
        st.session_state['last_status'] = last_status

    # Data freshness check
    if not st.session_state['sensor_data'].empty:
        latest_time = st.session_state['sensor_data']['timestamp'].max()
    else:
        latest_time = None

    if is_stale(latest_time):
        st.warning("⚠️ Data is stale or not received within 40 seconds. Check sensor connectivity.")

    # Machine status overview
    st.header("Machine Status Overview")
    data_latest = st.session_state['sensor_data'].copy()
    if not data_latest.empty:
        data_latest = data_latest.groupby(['machine', 'sensor']).apply(lambda g: g.sort_values('timestamp').iloc[-1]).reset_index(drop=True)
    else:
        data_latest = pd.DataFrame(columns=['timestamp','machine','sensor','value'])

    cols = st.columns(len(MACHINES))
    for i, machine in enumerate(MACHINES):
        with cols[i]:
            st.subheader(machine)
            for sensor in SENSORS:
                row = data_latest[(data_latest['machine']==machine) & (data_latest['sensor']==sensor)]
                val = row['value'].values[0] if not row.empty else None
                ts = row['timestamp'].values[0] if not row.empty else None
                status = classify_status(val, sensor, machine, st.session_state['thresholds'])
                color = STATUS_COLORS.get(status, 'grey')
                display_val = f"{val:.2f}" if val is not None and not pd.isna(val) else "--"
                ts_str = pd.to_datetime(ts).strftime("%Y-%m-%d %H:%M:%S") if ts is not None else "No timestamp"
                st.markdown(f"<div style='color:{color}; font-weight:bold;' title='Last updated: {ts_str}'>{sensor.capitalize()}: {display_val} ({status})</div>", unsafe_allow_html=True)
                simulate_alert(status, machine, sensor, display_val)
            plot_combined_health(st.session_state['sensor_data'], machine, st.session_state['thresholds'])

    # Trends & predictions
    st.header("Trends and Prediction")
    trend_machine = st.selectbox("Select Machine for Trend", MACHINES, key="trend_machine")
    trend_sensor = st.selectbox("Select Sensor for Trend", SENSORS, key="trend_sensor")

    recent_time = datetime.now() - timedelta(minutes=60)
    hist_df = st.session_state['sensor_data']
    hist_filtered = hist_df[
        (hist_df['timestamp'] >= recent_time) &
        (hist_df['machine'] == trend_machine) &
        (hist_df['sensor'] == trend_sensor)
    ].sort_values('timestamp')
    plot_history(hist_filtered, trend_machine, trend_sensor)

    if not hist_filtered.empty:
        warn, crit = st.session_state['thresholds'][trend_machine][trend_sensor]
        ttc = estimate_time_to_critical(hist_filtered['value'].values, hist_filtered['timestamp'], warn, crit)
        if ttc is not None:
            st.info(f"Estimated time to critical breach: {ttc}")

        slope = moving_average_slope(hist_filtered['value'].values, window=5)
        if slope is not None and slope > 0:
            pred_next = hist_filtered['value'].values[-1] + slope
            if warn <= pred_next < crit:
                st.warning(f"Predictive Alert: Sensor trending towards Warning level (approx. next reading: {pred_next:.2f})")
            elif pred_next >= crit:
                st.error(f"Predictive Alert: Sensor trending towards Critical level (approx. next reading: {pred_next:.2f})")

        explanation = llm_explain_trend(trend_machine, trend_sensor, hist_filtered)
        with st.expander("Explain Sensor Status / Trend (LLM):"):
            st.write(explanation)
    else:
        st.write("No historical data to display trends.")

    # Operator notes
    st.header("Operator Maintenance Log")
    notes = st.session_state['operator_notes']
    note_text = st.text_area(f"Notes for {trend_machine}", value=notes.get(trend_machine, ""), height=150)
    if st.button("Save Notes"):
        if note_text.strip():
            notes[trend_machine] = note_text.strip()
            save_operator_notes(notes)
            st.session_state['operator_notes'] = notes
            st.success("Notes saved.")
        else:
            st.warning("Cannot save empty notes.")

    # Alert logs for admin
    if role == 'admin':
        st.header("Alert Logs (Admin)")
        alert_log_text = load_alert_log()
        st.text_area("Alert Log", alert_log_text, height=200)

    # PDF export
    st.header("Export Report")
    if st.button(f"Export PDF Report - {trend_machine}"):
        pdf_bytes = export_pdf(trend_machine, st.session_state['sensor_data'], load_alert_log(), st.session_state['operator_notes'])
        href = get_pdf_download_link(pdf_bytes, trend_machine)
        st.markdown(href, unsafe_allow_html=True)

    # MQTT info (placeholder)
    st.sidebar.header("MQTT Simulation")
    mqtt_simulation_stub()

    st.markdown("---")
    st.caption("Developed for Workshop-Watch Hackathon Demo - Industrial IoT Real-Time Monitoring")


if __name__ == "__main__":
    main()
