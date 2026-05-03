import streamlit as st

st.set_page_config(page_title="Multi Object Tracking Demo", layout="wide")

st.title("Multi-Object Detection and Persistent ID Tracking in Public Sports Footage")

st.write("""
This project demonstrates an end-to-end computer vision pipeline for detecting multiple football players
and maintaining persistent identities across frames using YOLOv8s and BoT-SORT.
""")

st.subheader("Project Summary")

st.markdown("""
- **Detection Model Used:** YOLOv8s  
- **Tracking Algorithm Used:** BoT-SORT  
- **Public Source Video:** https://youtu.be/M9mKnmt0YaM?si=MmPbVWQH2FzXQRME  
- **Processed Segment:** 00:00:45 to 00:00:58  
- **Output:** Annotated MP4 video with bounding boxes, persistent player IDs, center points, and tracked player count.
""")

st.subheader("Final Annotated Output Video")

st.markdown("### Public Hosted Video Demo")
st.markdown("[Click Here to View Final Annotated Output Video](https://drive.google.com/file/d/1jnTccmztuLyuktLRA4Z5kyJJm5Wu_2-C/view?usp=sharing)")

st.subheader("Sample Result Screenshots")

col1, col2, col3 = st.columns(3)

with col1:
    st.image("frame_start.png", caption="Initial Multi-Player Detection", use_container_width=True)

with col2:
    st.image("frame_middle.png", caption="Mid Sequence Persistent Tracking", use_container_width=True)

with col3:
    st.image("frame_end.png", caption="Dense Player Clustering with ID Continuity", use_container_width=True)

st.subheader("Technical Highlights")

st.markdown("""
This implementation performs:

- frame-by-frame football player detection,
- persistent ID assignment across moving subjects,
- human-readable Player labels,
- center point movement visualization,
- final MP4 annotated output generation.

Despite practical sports-video challenges such as camera panning, player overlap, and partial occlusion,
the system maintains reasonably stable tracking identities across the selected gameplay sequence.
""")

st.success("Project Successfully Demonstrates Practical Multi-Object Persistent ID Tracking")