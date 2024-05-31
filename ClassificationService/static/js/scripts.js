document.querySelector('.overlay').style.display = 'none';
// Функция для получения данных с сервера
function showData(request) {
    var xhr = new XMLHttpRequest();
    xhr.onreadystatechange = function() {
        if (xhr.readyState == 4 && xhr.status == 200) {
            var data = JSON.parse(xhr.responseText);
            fillTableWithData(data); // Вызываем функцию для заполнения таблицы данными
        }
    };
    xhr.open("GET", request, true);
    xhr.send();
}

// Функция для заполнения таблицы данными
function fillTableWithData(data) {
    var tableBody = document.getElementById("data-body");
    tableBody.innerHTML = ""; // Очищаем содержимое тела таблицы

    // Проходим по данным и добавляем строки в таблицу
    for (var i = 0; i < data.length; i++) {
        var row = "<tr>";
        row += "<td class=\"id-" + data[i].type + "\">" + data[i].id + "</td>";
        row += "<td class=\"name\">" + data[i].name + "</td>";
        row += "</tr>";
        tableBody.innerHTML += row;
    }
}
// При нажатии на кнопку показываем окошко выбора файла
document.getElementById('fileUploadButton').addEventListener('click', function() {
    document.getElementById('fileInput').click();
});
// Обработка выбора файла и его загрузка на сервер
document.getElementById('fileInput').addEventListener('change', function() {
    var file = this.files[0]; // Получаем выбранный файл
    var formData = new FormData(); // Создаем объект FormData для передачи файла на сервер
    formData.append('file', file); // Добавляем файл в объект FormData под ключом 'file'

    // Создаем объект XMLHttpRequest для отправки запроса на сервер
    var xhr = new XMLHttpRequest();
    xhr.open('POST', '/add_ethalon', true); // Открываем POST запрос на адрес '/upload'

    // Обработка завершения запроса
    xhr.onload = function() {
        if (xhr.status === 200) {
            alert(xhr.responseText);
        } else {
            alert(xhr.responseText);
        }
    };

    // Отправляем запрос на сервер с данными файла
    xhr.send(formData);
});

// Отправить данные для получения топ-N эталонов
function sendRankingData() {
    // Показать оверлей
    document.querySelector('.overlay').style.display = 'flex';

    var formData = new FormData(document.getElementById("rankingForm"));
    var xhr = new XMLHttpRequest();
    xhr.onreadystatechange = function() {
        if (xhr.readyState === XMLHttpRequest.DONE) {
            if (xhr.status === 200) {
                var response = JSON.parse(xhr.responseText);
                displayResults(response);
            } else {
                alert("Ошибка: " + xhr.status);
            }
        }
    };
    xhr.open("POST", "/get_top_etalons", true);
    xhr.send(formData);
}

// Отображение результатов ранжирования в таблице
function displayResults(etalons) {
    var tableBody = document.getElementById("data-body");
    // Очистка контейнера результатов
    tableBody.innerHTML = "";

    // Проходимся по объекту rang и добавляем строки в таблицу
    for (var position in etalons) {
        if (etalons.hasOwnProperty(position)) {
            var jobTitles = etalons[position];

            // Создаем строку таблицы для каждой должности и её массива должностей
            var row = "<tr>";
            row += "<td>" + position + "</td>"; // Должность
            row += "<td>" + jobTitles.join("; ") + "</td>"; // Массив должностей
            row += "</tr>";

            // Добавляем строку в таблицу
            tableBody.innerHTML += row;
        }
    }
    // Скрыть оверлей после завершения анимации или выполнения задачи
    document.querySelector('.overlay').style.display = 'none';
}

// Обработчик клика на таблице
document.getElementById('data-table').addEventListener('click', function(event) {
    var target = event.target; // Получаем элемент, на котором произошло событие клика
    if (target.tagName === 'TD' && target.className === 'name') { // Проверяем, что клик произошел на ячейке с классом 'name-cell'
        var id = target.parentNode.cells[0].textContent; // Получаем значение ID из первой ячейки строки
        var className = target.parentNode.cells[0].getAttribute('class');
        // Отправляем запрос на сервер для получения дополнительной информации о строке по id
        fetch('/get_row_info?id=' + id + '&type=' + className)
            .then(function(response) {
                return response.json();
            })
            .then(function(data) {
                // Открываем модальное окно и выводим полученные данные
                showModal(data);
            })
            .catch(function(error) {
                console.error('Ошибка при получении данных:', error);
            });
    }
});

// Функция для отображения модального окна с описанием эталона
function showModal(description) {
    var modal = document.getElementById("modal");
    var modalDescription = document.getElementById("modal-description");

    modalDescription.textContent = description; // Устанавливаем описание эталона в модальное окно
    modal.style.display = "block"; // Показываем модальное окно
}

function closeModal() {
    var modal = document.getElementById("modal");
    modal.style.display = "none"; // Скрываем модальное окно
}

// Добавляем обработчик клика на кнопку "Закрыть"
var closeButton = document.querySelector(".close");
closeButton.addEventListener("click", closeModal)

// Закрытие модального окна при клике вне его области
window.addEventListener("click", function(event) {
    var modal = document.getElementById("modal");
    if (event.target == modal) {
        modal.style.display = "none";
    }
});