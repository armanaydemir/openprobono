async () => {

    // GTM scripts
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

    // dynamically resize user input textbox
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

    // example prompt modal HTML and logic
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
    h2.innerHTML = "Choose a prompt to paste it in the text field";
    modalContent.appendChild(h2);

    var categories = {
        "Criminal Law": [
            "What are my rights if I'm arrested?",
            "Is it legal to record a conversation without consent?",
            "What is the difference between assault and battery?",
            "Can you explain the process of plea bargaining?"
        ],
        "Family Law": [
            "How is child custody determined during a divorce?",
            "What are the legal requirements for getting a restraining order?",
            "What's the process for adoption in my state?",
            "How does spousal support work?"
        ],
        "Employment Law": [
            "Can my employer fire me without cause?",
            "What should I do if I'm facing workplace discrimination?",
            "What are the wage and hour laws in my jurisdiction?",
            "How do I negotiate a fair employment contract?"
        ],
        "Intellectual Property Law": [
            "How do I copyright my creative work?",
            "What is the process for filing a patent?",
            "What constitutes fair use in copyright law?",
            "How can I protect my company's trademarks?"
        ],
        "Real Estate Law": [
            "What's the process for buying a home and closing a deal?",
            "Can you explain zoning laws and their impact on property use?",
            "What are my rights and responsibilities as a tenant?",
            "How do easements work in real estate?"
        ],
        "Personal Injury Law": [
            "What should I do if I've been injured in a car accident?",
            "How can I prove liability in a personal injury case?",
            "What damages can I claim in a personal injury lawsuit?",
            "What is the statute of limitations for personal injury claims?"
        ],
        "Immigration Law": [
            "What are the different types of U.S. visas available?",
            "How does the naturalization process work for permanent residents?",
            "Can you explain the asylum application process?",
            "What are the consequences of overstaying a visa?"
        ]
    };

    for (var category in categories) {
        var categoryHeading = document.createElement("button");
        categoryHeading.className = "accordion";
        categoryHeading.innerHTML = category;

        var categoryPanel = document.createElement("div");
        categoryPanel.className = "panel";

        modalContent.appendChild(categoryHeading);
        modalContent.appendChild(categoryPanel);

        var categoryQuestions = categories[category];
        for (var i = 0; i < categoryQuestions.length; i++) {
            var question = document.createElement("p");
            question.className = "example-prompt";
            question.innerHTML = categoryQuestions[i];
            question.addEventListener("click", function() {
                var selectedQuestion = this.innerHTML;
                var textInput = document.querySelector("#userquery textarea");
                textInput.value = selectedQuestion;
                closePopup.click();  // Close the popup after selecting a question
            });
            categoryPanel.appendChild(question);
        }
    }

    popupContainer.appendChild(modalContent);
    document.body.appendChild(popupContainer);
    
    const scriptModal = document.createElement("script");
    scriptModal.innerHTML = `
        var popupContainer = document.getElementById("popup-container");
        var closePopup = document.getElementById("close-popup");

        // close button
        closePopup.addEventListener("click", function() {
            popupContainer.style.display = "none";
        });

        // open button
        var promptButton = document.getElementById("open-popup");
        promptButton.addEventListener("click", function() {
            if (popupContainer.style.display == "block")
                popupContainer.style.display = "none";
            else
                popupContainer.style.display = "block";
        });

        // expand/collapse accordions
        var acc = document.getElementsByClassName("accordion");
        var i;
        for (i = 0; i < acc.length; i++) {
            acc[i].addEventListener("click", function() {
                this.classList.toggle("active");
                var panel = this.nextElementSibling;
                if (panel.style.maxHeight) {
                panel.style.maxHeight = null;
                } else {
                panel.style.maxHeight = panel.scrollHeight + "px";
                } 
            });
        }
    `;
    document.head.appendChild(scriptModal);    
}