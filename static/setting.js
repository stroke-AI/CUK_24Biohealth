document.addEventListener("DOMContentLoaded", function() {
    // 플래시 메시지 처리
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

    // 혈압 입력 필드 제어
    function toggleBloodPressureInput() {
        const bpOption = document.getElementById("blood-pressure-option").value;
        const bpInput = document.getElementById("blood-pressure-input");
        bpInput.style.display = (bpOption === "unknown") ? "none" : "block";
    }

    // 콜레스테롤 입력 필드 제어
    function toggleCholesterolInput() {
        const cholOption = document.getElementById("cholesterol-option").value;
        const cholInput = document.getElementById("cholesterol-input");
        cholInput.style.display = (cholOption === "unknown") ? "none" : "block";
    }

    // 이벤트 리스너 추가
    document.getElementById("blood-pressure-option").addEventListener("change", toggleBloodPressureInput);
    document.getElementById("cholesterol-option").addEventListener("change", toggleCholesterolInput);

    // 초기 상태 설정
    toggleBloodPressureInput();
    toggleCholesterolInput();
});
