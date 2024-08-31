document.addEventListener('DOMContentLoaded', function() {
    const flashes = document.querySelectorAll('.flashes li');
    flashes.forEach(function(flash) {
        const message = flash.textContent;
        const category = flash.className;
        if (category === 'success') {
            alert("성공: " + message);
        } else if (category === 'error') {
            alert("오류: " + message);
        } else if (category === 'info') {
            alert("정보: " + message);
        }
    });
});
