const video = document.getElementById('video');
const captureBtn = document.getElementById('capture');
const resultDiv = document.getElementById('result');

// Akses webcam
navigator.mediaDevices.getUserMedia({ video: true })
.then(stream => {
    video.srcObject = stream;
});

captureBtn.addEventListener('click', () => {
    const canvas = document.createElement('canvas');
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    const ctx = canvas.getContext('2d');
    ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
    
    canvas.toBlob(blob => {
        const reader = new FileReader();
        reader.onloadend = () => {
            const arrayBuffer = reader.result;
            fetch('/detect', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({ image: Array.from(new Uint8Array(arrayBuffer)) })
            })
            .then(res => res.json())
            .then(data => {
                if(data.status === 'found'){
                    resultDiv.innerHTML = `Wajah Dikenali: ${data.name}`;
                } else {
                    resultDiv.innerHTML = 'Wajah Tidak Dikenali';
                }
            });
        };
        reader.readAsArrayBuffer(blob);
    }, 'image/jpeg');
});
