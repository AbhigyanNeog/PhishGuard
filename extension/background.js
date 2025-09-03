// background.js
// Simple rule-based detection
function isSuspicious(url) {
    if (url.length > 75) return true;
    if (url.includes('@')) return true;
    if (url.includes('-')) return true;
    if (!url.startsWith("https")) return true;
    return false;
}

// Trigger before navigation
chrome.webNavigation.onBeforeNavigate.addListener(function(details) {
    let url = details.url;

    if (isSuspicious(url)) {
        chrome.notifications.create({
            type: "basic",
            iconUrl: "icon.png",
            title: "⚠️ PhishGuard Alert",
            message: "This site may be a phishing attempt!\n" + url
        });
    }
}, {url: [{schemes: ["http", "https"]}]});
