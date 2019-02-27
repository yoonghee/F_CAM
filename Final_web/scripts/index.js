<script src="https://cdnjs.com/libraries/bodymovin"https://cdnjs.cloudflare.com/ajax/libs/bodymovin/5.4.3/lottie_canvas.js type="text/javascript"></script>
var animation = bodymovin.loadAnimation({
    container: document.getElementById('bm'),
    renderer: 'svg',
    loop: true,
    autoplay: true,
    path: '../data.json'
  })