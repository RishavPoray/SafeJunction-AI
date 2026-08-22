import streamlit as st
from ultralytics import YOLO
import tempfile
import os
import subprocess
import imageio_ffmpeg
import cv2
import math
import uuid
from collections import defaultdict


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="SafeJunction AI",
    page_icon="🚦",
    layout="wide"
)


# ============================================================
# CUSTOM UI
# ============================================================

st.markdown("""
<style>

.main-title {
    font-size: 42px;
    font-weight: 800;
    margin-bottom: 0;
}

.subtitle {
    font-size: 18px;
    color: #777;
    margin-bottom: 25px;
}

.metric-card {
    padding: 20px;
    border-radius: 12px;
    border: 1px solid #ddd;
    text-align: center;
}

.metric-number {
    font-size: 32px;
    font-weight: 700;
}

.metric-label {
    font-size: 15px;
}

.section-title {
    font-size: 26px;
    font-weight: 700;
}

</style>
""", unsafe_allow_html=True)


# ============================================================
# HEADER
# ============================================================

st.markdown(
    '<div class="main-title">🚦 SafeJunction AI</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'AI-Powered Traffic Junction Safety & Risk Monitoring'
    '</div>',
    unsafe_allow_html=True
)

st.divider()


# ============================================================
# UPLOAD
# ============================================================

st.header("📹 Traffic Video Analysis")

uploaded_video = st.file_uploader(
    "Upload a traffic-junction video",
    type=["mp4", "avi", "mov", "mkv"]
)


if uploaded_video:

    st.success(
        f"Uploaded: **{uploaded_video.name}** ✅"
    )

    st.video(uploaded_video)

    if st.button(
        "🤖 Analyze Traffic",
        use_container_width=True
    ):

        # ====================================================
        # UNIQUE SESSION
        # ====================================================

        session_id = uuid.uuid4().hex

        session_dir = os.path.join(
            tempfile.gettempdir(),
            "SafeJunctionAI",
            session_id
        )

        os.makedirs(
            session_dir,
            exist_ok=True
        )

        input_path = os.path.join(
            session_dir,
            "input.mp4"
        )

        raw_output = os.path.join(
            session_dir,
            "tracking.mp4"
        )

        final_output = os.path.join(
            session_dir,
            "SafeJunction_Result.mp4"
        )


        with open(
            input_path,
            "wb"
        ) as f:

            f.write(
                uploaded_video.getbuffer()
            )


        # ====================================================
        # LOAD YOLO
        # ====================================================

        st.info(
            "Loading YOLO AI model..."
        )

        model = YOLO(
            "yolo11n.pt"
        )

        st.success(
            "YOLO AI model loaded! ✅"
        )


        # ====================================================
        # VIDEO INFO
        # ====================================================

        cap = cv2.VideoCapture(
            input_path
        )

        fps = cap.get(
            cv2.CAP_PROP_FPS
        )

        width = int(
            cap.get(
                cv2.CAP_PROP_FRAME_WIDTH
            )
        )

        height = int(
            cap.get(
                cv2.CAP_PROP_FRAME_HEIGHT
            )
        )

        cap.release()


        if fps <= 0:
            fps = 30


        # ====================================================
        # VIDEO WRITER
        # ====================================================

        fourcc = cv2.VideoWriter_fourcc(
            *"mp4v"
        )

        writer = cv2.VideoWriter(
            raw_output,
            fourcc,
            fps,
            (width, height)
        )


        # ====================================================
        # TRACKING DATA
        # ====================================================

        unique_objects = defaultdict(set)

        pair_close_frames = defaultdict(int)

        persistent_interactions = set()

        interaction_locations = {}

        interaction_distance = 100

        persistence_threshold = 8

        frame_count = 0


        # ====================================================
        # YOLO TRACKING
        # ====================================================

        st.info(
            "🤖 AI is detecting and tracking objects..."
        )

        results = model.track(
            source=input_path,
            conf=0.4,
            persist=True,
            stream=True,
            verbose=False
        )


        # ====================================================
        # FRAME PROCESSING
        # ====================================================

        for result in results:

            frame_count += 1

            frame = result.orig_img.copy()

            pedestrians = {}

            vehicles = {}


            if result.boxes is not None:

                ids = (
                    result.boxes.id.cpu().tolist()
                    if result.boxes.id is not None
                    else []
                )

                classes = (
                    result.boxes.cls.cpu().tolist()
                )

                boxes = (
                    result.boxes.xyxy.cpu().tolist()
                )


                for track_id, class_id, box in zip(
                    ids,
                    classes,
                    boxes
                ):

                    track_id = int(
                        track_id
                    )

                    class_id = int(
                        class_id
                    )

                    name = model.names[
                        class_id
                    ]

                    unique_objects[
                        name
                    ].add(
                        track_id
                    )


                    x1, y1, x2, y2 = map(
                        int,
                        box
                    )

                    cx = (
                        x1 + x2
                    ) // 2

                    cy = (
                        y1 + y2
                    ) // 2


                    if name == "person":

                        pedestrians[
                            track_id
                        ] = (
                            cx,
                            cy
                        )


                    elif name in [
                        "car",
                        "motorcycle",
                        "bus",
                        "truck"
                    ]:

                        vehicles[
                            track_id
                        ] = (
                            cx,
                            cy
                        )


                    # Draw detection box

                    cv2.rectangle(
                        frame,
                        (x1, y1),
                        (x2, y2),
                        (0, 255, 0),
                        2
                    )

                    cv2.putText(
                        frame,
                        f"{name} ID:{track_id}",
                        (
                            x1,
                            max(
                                20,
                                y1 - 8
                            )
                        ),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.55,
                        (0, 255, 0),
                        2
                    )


            # =================================================
            # INTERACTION DETECTION
            # =================================================

            current_pairs = set()


            for pid, person in (
                pedestrians.items()
            ):

                for vid, vehicle in (
                    vehicles.items()
                ):

                    distance = math.sqrt(
                        (
                            person[0]
                            -
                            vehicle[0]
                        ) ** 2
                        +
                        (
                            person[1]
                            -
                            vehicle[1]
                        ) ** 2
                    )


                    pair = (
                        pid,
                        vid
                    )


                    if distance < interaction_distance:

                        current_pairs.add(
                            pair
                        )

                        pair_close_frames[
                            pair
                        ] += 1

                    else:

                        pair_close_frames[
                            pair
                        ] = 0


                    if (
                        pair_close_frames[pair]
                        >= persistence_threshold
                    ):

                        persistent_interactions.add(
                            pair
                        )

                        interaction_locations[
                            pair
                        ] = (
                            int(
                                (
                                    person[0]
                                    +
                                    vehicle[0]
                                ) / 2
                            ),
                            int(
                                (
                                    person[1]
                                    +
                                    vehicle[1]
                                ) / 2
                            )
                        )


            # =================================================
            # HOTSPOT MARKER
            # =================================================

            for pair in persistent_interactions:

                if pair in interaction_locations:

                    hx, hy = (
                        interaction_locations[
                            pair
                        ]
                    )


                    cv2.circle(
                        frame,
                        (hx, hy),
                        45,
                        (0, 0, 255),
                        4
                    )


                    cv2.circle(
                        frame,
                        (hx, hy),
                        8,
                        (0, 0, 255),
                        -1
                    )


                    cv2.putText(
                        frame,
                        "RISK HOTSPOT",
                        (
                            max(
                                10,
                                hx - 80
                            ),
                            max(
                                30,
                                hy - 55
                            )
                        ),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (0, 0, 255),
                        2
                    )


            writer.write(
                frame
            )


        writer.release()

        st.success(
            "AI tracking completed! ✅"
        )


        # ====================================================
        # COUNTS
        # ====================================================

        people = len(
            unique_objects["person"]
        )

        cars = len(
            unique_objects["car"]
        )

        motorcycles = len(
            unique_objects["motorcycle"]
        )

        buses = len(
            unique_objects["bus"]
        )

        trucks = len(
            unique_objects["truck"]
        )

        total_vehicles = (
            cars
            +
            motorcycles
            +
            buses
            +
            trucks
        )

        heavy_vehicles = (
            buses
            +
            trucks
        )

        interactions = len(
            persistent_interactions
        )


        # ====================================================
        # DENSITY
        # ====================================================

        if total_vehicles <= 20:

            density = "LOW"
            density_icon = "🟢"

        elif total_vehicles <= 50:

            density = "MODERATE"
            density_icon = "🟡"

        else:

            density = "HIGH"
            density_icon = "🔴"


        # ====================================================
        # SAFETY SCORE
        # ====================================================

        score = 100

        breakdown = []


        if density == "MODERATE":

            score -= 10

            breakdown.append(
                "Traffic congestion: -10"
            )

        elif density == "HIGH":

            score -= 15

            breakdown.append(
                "High traffic density: -15"
            )


        if people > 30:

            score -= 5

            breakdown.append(
                "High pedestrian activity: -5"
            )

        elif people > 10:

            score -= 2

            breakdown.append(
                "Moderate pedestrian activity: -2"
            )


        if heavy_vehicles >= 20:

            score -= 7

            breakdown.append(
                "High heavy-vehicle activity: -7"
            )

        elif heavy_vehicles >= 10:

            score -= 4

            breakdown.append(
                "Moderate heavy-vehicle activity: -4"
            )


        if interactions > 0:

            penalty = (
                8
                if interactions <= 5
                else 15
            )

            score -= penalty

            breakdown.append(
                "Persistent pedestrian-vehicle "
                f"interactions: -{penalty}"
            )


        score = max(
            0,
            min(
                100,
                score
            )
        )


        # ====================================================
        # RISK
        # ====================================================

        if score >= 80:

            risk = "LOW"
            risk_icon = "🟢"

        elif score >= 60:

            risk = "MODERATE"
            risk_icon = "🟡"

        elif score >= 40:

            risk = "ELEVATED"
            risk_icon = "🟠"

        else:

            risk = "HIGH"
            risk_icon = "🔴"


        # ====================================================
        # DASHBOARD
        # ====================================================

        st.divider()

        st.markdown(
            '<div class="section-title">'
            '📊 Traffic Safety Dashboard'
            '</div>',
            unsafe_allow_html=True
        )

        st.write("")


        # ====================================================
        # TOP METRICS
        # ====================================================

        c1, c2, c3, c4 = st.columns(4)


        with c1:

            st.metric(
                "🚗 Vehicles",
                total_vehicles
            )


        with c2:

            st.metric(
                "🚶 People",
                people
            )


        with c3:

            st.metric(
                "⚠️ Risk Events",
                interactions
            )


        with c4:

            st.metric(
                "🛡️ Safety Score",
                f"{score}/100"
            )


        st.divider()


        # ====================================================
        # STATUS CARDS
        # ====================================================

        left, right = st.columns(2)


        with left:

            st.subheader(
                "🚦 Traffic Status"
            )

            st.markdown(
                f"# {density_icon} {density}"
            )

            st.write(
                f"{total_vehicles} unique vehicles detected."
            )


        with right:

            st.subheader(
                "⚠️ Overall Safety"
            )

            st.markdown(
                f"# {risk_icon} {risk}"
            )

            st.progress(
                score / 100
            )


        # ====================================================
        # DETECTION DETAILS
        # ====================================================

        st.divider()

        st.subheader(
            "🤖 AI Detection Results"
        )


        d1, d2, d3, d4, d5 = (
            st.columns(5)
        )


        d1.metric(
            "🚶 People",
            people
        )

        d2.metric(
            "🚗 Cars",
            cars
        )

        d3.metric(
            "🏍️ Motorcycles",
            motorcycles
        )

        d4.metric(
            "🚌 Buses",
            buses
        )

        d5.metric(
            "🚚 Trucks",
            trucks
        )


        # ====================================================
        # SAFETY ALERT
        # ====================================================

        st.divider()

        st.subheader(
            "🚨 Safety Alert"
        )


        if interactions > 0:

            st.warning(
                "Potential pedestrian-vehicle "
                "conflict detected. "
                "Monitoring recommended."
            )

        else:

            st.success(
                "No persistent pedestrian-vehicle "
                "conflicts detected."
            )


        # ====================================================
        # HOTSPOT
        # ====================================================

        st.subheader(
            "🗺️ Risk Hotspot"
        )


        if interactions > 0:

            first_pair = next(
                iter(
                    interaction_locations
                )
            )

            hx, hy = (
                interaction_locations[
                    first_pair
                ]
            )


            st.warning(
                f"{interactions} risk hotspot(s) detected."
            )

            st.write(
                f"Approximate location: "
                f"X: {hx}px | Y: {hy}px"
            )

            st.info(
                "🔴 Risk hotspot is marked "
                "directly in the AI tracking video."
            )

        else:

            st.success(
                "No risk hotspots detected."
            )


        # ====================================================
        # RISK BREAKDOWN
        # ====================================================

        st.subheader(
            "📋 Risk Breakdown"
        )


        if breakdown:

            for item in breakdown:

                st.write(
                    f"• {item}"
                )

        else:

            st.success(
                "No significant risk factors detected."
            )


        # ====================================================
        # SUMMARY
        # ====================================================

        st.divider()

        st.subheader(
            "📊 Detection Summary"
        )


        s1, s2, s3 = st.columns(3)


        s1.metric(
            "Unique Vehicles",
            total_vehicles
        )

        s2.metric(
            "Unique Pedestrians",
            people
        )

        s3.metric(
            "Heavy Vehicles",
            heavy_vehicles
        )


        st.write(
            f"Persistent pedestrian-vehicle "
            f"interactions: **{interactions}**"
        )

        st.write(
            f"Frames processed: **{frame_count}**"
        )


        # ====================================================
        # VIDEO CONVERSION
        # ====================================================

        st.info(
            "🎞️ Converting AI video to "
            "browser-compatible format..."
        )


        ffmpeg = (
            imageio_ffmpeg.get_ffmpeg_exe()
        )


        command = [
            ffmpeg,
            "-y",
            "-i",
            raw_output,
            "-c:v",
            "libx264",
            "-preset",
            "fast",
            "-crf",
            "23",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            "-an",
            final_output
        ]


        result = subprocess.run(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )


        # ====================================================
        # SHOW RESULT
        # ====================================================

        if (
            result.returncode == 0
            and os.path.exists(
                final_output
            )
        ):

            st.success(
                "Video conversion completed! ✅"
            )

            st.divider()

            st.header(
                "🎥 AI Tracking Result"
            )

            st.caption(
                f"Processed video: "
                f"{uploaded_video.name}"
            )


            with open(
                final_output,
                "rb"
            ) as f:

                st.video(
                    f.read()
                )


            st.success(
                "✅ Bounding boxes, tracking IDs "
                "and risk hotspot markers added."
            )

        else:

            st.error(
                "❌ Video conversion failed."
            )

            st.code(
                result.stderr[-3000:]
            )