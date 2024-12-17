const video = document.getElementById("camera");
const overlay = document.getElementById("camera-overlay");
const personLogs = document.getElementById('person-logs');
const toggleButton = document.getElementById("toggle-process");
const recognizerSelect = document.getElementById("select-recognizer");
const resetTrackerButton = document.getElementById("reset-tracker");

let processing = false;
let intervalId;
let isFetching = false;  // imitate threading lock
let currentAbortController;
const captureInterval = 350;

// Create a canvas for capturing frames (only used for Data URL conversion)
const canvas = document.createElement("canvas");
const context = canvas.getContext("2d");

async function startVideoStream() {
    try {
        const stream = await navigator.mediaDevices.getUserMedia({ video: true });
        video.srcObject = stream;
        video.play();
        video.onloadeddata = () => {
            canvas.width = video.videoWidth;
            canvas.height = video.videoHeight;
        };
    } catch (error) {
        alert("Unable to access the camera. Please check the console for detailed error.");
        console.error(error);
    }
}

async function captureFrame() {
    if (isFetching) {
        return;
    }

    isFetching = true;
    context.drawImage(video, 0, 0, canvas.width, canvas.height);
    const frame = canvas.toDataURL("image/jpeg");

    currentAbortController = new AbortController();
    const { signal } = currentAbortController;

    try {
        const response = await fetch("/process_frame", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ 
                image: frame,
                recognizer: recognizerSelect.value,
            }),
            signal: signal,
        });

        const data = await response.json();
        if (data.overlay) {
            overlay.src = data.overlay;
        }
    } catch (error) {
        if (error.name === "AbortError") {
            console.log("Fetch aborted");
        } else {
            console.error("Error sending frame to server:", error);
        }
    } finally {
        isFetching = false;
    }
}

async function fetchLogs() {
    await fetch("/tracker_logging")
        .then(response => response.json())
        .then(data => {

            data.sort((a, b) => new Date(b.last_seen) - new Date(a.last_seen));

            personLogs.innerHTML = ""; // Clear current logs
            data.forEach(person => {
                const logEntry = document.createElement("div");
                logEntry.classList.add("log-entry");
                logEntry.innerHTML = `
                    <strong>ID:</strong> ${person.id} <br>
                    <strong>Name:</strong> ${person.name} <br>
                    <strong>Last Seen:</strong> ${new Date(person.last_seen).toLocaleString()} <br>
                    <strong>Confidence:</strong> ${Math.round(person.confidence * 10000) / 100}% <br>
                    <strong>Last Recognized:</strong> ${new Date(person.last_recognized).toLocaleString()} <br>
                `;
                personLogs.appendChild(logEntry);
            });
        })
        .catch(error => console.error("Error fetching logs:", error));
}


toggleButton.addEventListener("click", () => {
    processing = !processing;
    toggleButton.textContent = processing ? "Stop Processing" : "Start Processing";

    if (processing) {
        intervalId = setInterval(captureFrame, captureInterval);
        setInterval(fetchLogs, 2000);
    } else {
        clearInterval(intervalId);
        intervalId = null;
        overlay.src = "";  // Clear the overlay image when stopped
    }

    if (currentAbortController) {
        currentAbortController.abort();
        currentAbortController = null;
    }
});

resetTrackerButton.addEventListener("click", async () => {
    if (processing) {
        alert("Please disable processing");
        return;
    }

    try {
        const _ = await fetch("/reset_tracker", {
            method: "DELETE",
            headers: { "Content-Type": "application/json" },
        });
    } catch (error) {
        console.error("Error sending frame to server:", error);
    }
})

startVideoStream();
