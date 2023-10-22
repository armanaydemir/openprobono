async () => {
    const scriptLoadGtm = document.createElement("script");
    scriptLoadGtm.onload = () => console.log("tag manager loaded") ;
    scriptLoadGtm.src = "https://www.googletagmanager.com/gtag/js?id=G-MKDNM9G2PQ";
    document.head.appendChild(scriptLoadGtm);

    const scriptRunGtm = document.createElement("script");
    scriptRunGtm.onload = () => {
        window.dataLayer = window.dataLayer || [];
        function gtag(){dataLayer.push(arguments);}
        gtag('js', new Date());
        gtag('config', 'G-MKDNM9G2PQ');
    }
    document.head.appendChild(scriptRunGtm);

    const scriptResizeTextbox = document.createElement("script");
    scriptResizeTextbox.innerHTML = `
        textarea = document.querySelector("#userquery textarea");
        textarea.addEventListener('input', autoResize, false);
 
        function autoResize() {
            this.style.height = 'auto';
            this.style.height = this.scrollHeight + 'px';
        }
    `;
    document.head.appendChild(scriptResizeTextbox);

    var popupContainer = document.createElement("div");
    popupContainer.className = "modal";
    popupContainer.id = "popup-container";
    popupContainer.style.display = "none";

    var modalContent = document.createElement("div");
    modalContent.className = "modal-content";

    var closeSpan = document.createElement("span");
    closeSpan.className = "close";
    closeSpan.id = "close-popup";
    closeSpan.innerHTML = "&times;";
    modalContent.appendChild(closeSpan);

    var h2 = document.createElement("h2");
    h2.innerHTML = "Select a Question";
    modalContent.appendChild(h2);

    var categories = {
        "Criminal Law": [
            "What are my rights if I'm arrested?",
            "Is it legal to record a conversation without consent?"
            // Add more questions for this category...
        ],
        // Add more categories and questions here...
    };

    for (var category in categories) {
        var categoryHeading = document.createElement("h3");
        categoryHeading.innerHTML = category;
        modalContent.appendChild(categoryHeading);

        var categoryQuestions = categories[category];
        for (var i = 0; i < categoryQuestions.length; i++) {
            var question = document.createElement("p");
            question.className = "question";
            question.innerHTML = categoryQuestions[i];
            question.addEventListener("click", function() {
                var selectedQuestion = this.innerHTML;
                var textInput = document.querySelector("#userquery textarea");
                textInput.value = selectedQuestion;
                closePopup.click();  // Close the popup after selecting a question
            });
            modalContent.appendChild(question);
        }
    }

    popupContainer.appendChild(modalContent);
    document.body.appendChild(popupContainer);
    
    const scriptModal = document.createElement("script");
    scriptModal.innerHTML = `
        var popupContainer = document.getElementById("popup-container");
        var closePopup = document.getElementById("close-popup");

        closePopup.addEventListener("click", function() {
            popupContainer.style.display = "none";
        });

        var promptButton = document.getElementById("open-popup");
        promptButton.addEventListener("click", function() {
            if (popupContainer.style.display == "block")
                popupContainer.style.display = "none";
            else
                popupContainer.style.display = "block";
        });
    `;
    document.head.appendChild(scriptModal);    
}