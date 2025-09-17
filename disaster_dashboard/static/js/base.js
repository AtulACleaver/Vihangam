// Base JavaScript for Vihangam Drone Disaster Management System

// Global utility functions and common functionality

document.addEventListener('DOMContentLoaded', function() {
    console.log('Vihangam Disaster Management System Loaded');
    
    // Initialize common features
    initializeTooltips();
    initializeAlerts();
    setupNavigationHighlighting();
});

// Initialize Bootstrap tooltips
function initializeTooltips() {
    var tooltipTriggerList = [].slice.call(document.querySelectorAll('[data-bs-toggle="tooltip"]'));
    var tooltipList = tooltipTriggerList.map(function (tooltipTriggerEl) {
        return new bootstrap.Tooltip(tooltipTriggerEl);
    });
}

// Initialize alert auto-dismiss
function initializeAlerts() {
    // Auto-dismiss alerts after 5 seconds
    const alerts = document.querySelectorAll('.alert:not(.alert-permanent)');
    alerts.forEach(function(alert) {
        if (!alert.classList.contains('alert-permanent')) {
            setTimeout(function() {
                const alertInstance = new bootstrap.Alert(alert);
                alertInstance.close();
            }, 5000);
        }
    });
}

// Setup navigation highlighting
function setupNavigationHighlighting() {
    const currentPath = window.location.pathname;
    const navLinks = document.querySelectorAll('.navbar-nav .nav-link');
    
    navLinks.forEach(function(link) {
        if (link.getAttribute('href') === currentPath) {
            link.classList.add('active');
        }
    });
}

// Utility function to format time
function formatTime(date) {
    return date.toLocaleTimeString('en-US', {
        hour12: false,
        hour: '2-digit',
        minute: '2-digit',
        second: '2-digit'
    });
}

// Utility function to format date
function formatDate(date) {
    return date.toLocaleDateString('en-US', {
        year: 'numeric',
        month: 'short',
        day: 'numeric'
    });
}

// Show notification function
function showNotification(message, type = 'info', duration = 3000) {
    // Create notification element
    const notification = document.createElement('div');
    notification.className = `alert alert-${type} alert-dismissible fade show position-fixed`;
    notification.style.cssText = 'top: 20px; right: 20px; z-index: 9999; min-width: 300px;';
    
    notification.innerHTML = `
        <i class="fas fa-${getIconForType(type)} me-2"></i>
        ${message}
        <button type="button" class="btn-close" data-bs-dismiss="alert"></button>
    `;
    
    // Add to DOM
    document.body.appendChild(notification);
    
    // Auto remove after duration
    setTimeout(function() {
        const alertInstance = new bootstrap.Alert(notification);
        alertInstance.close();
    }, duration);
}

// Get icon for notification type
function getIconForType(type) {
    const icons = {
        'success': 'check-circle',
        'danger': 'exclamation-triangle',
        'warning': 'exclamation-circle',
        'info': 'info-circle',
        'primary': 'bell'
    };
    return icons[type] || 'info-circle';
}

// Loading state handler
function setLoadingState(element, loading = true) {
    if (loading) {
        element.disabled = true;
        const originalText = element.textContent;
        element.setAttribute('data-original-text', originalText);
        element.innerHTML = '<i class="fas fa-spinner fa-spin me-2"></i>Loading...';
    } else {
        element.disabled = false;
        const originalText = element.getAttribute('data-original-text');
        element.textContent = originalText;
    }
}

// WebSocket connection handler (placeholder for future implementation)
class VihangamWebSocket {
    constructor(url) {
        this.url = url;
        this.socket = null;
        this.reconnectAttempts = 0;
        this.maxReconnectAttempts = 5;
    }
    
    connect() {
        try {
            this.socket = new WebSocket(this.url);
            
            this.socket.onopen = () => {
                console.log('WebSocket connected');
                this.reconnectAttempts = 0;
                this.onConnectionOpen();
            };
            
            this.socket.onmessage = (event) => {
                const data = JSON.parse(event.data);
                this.handleMessage(data);
            };
            
            this.socket.onclose = () => {
                console.log('WebSocket disconnected');
                this.handleReconnect();
            };
            
            this.socket.onerror = (error) => {
                console.error('WebSocket error:', error);
            };
        } catch (error) {
            console.error('Failed to create WebSocket connection:', error);
        }
    }
    
    handleReconnect() {
        if (this.reconnectAttempts < this.maxReconnectAttempts) {
            this.reconnectAttempts++;
            setTimeout(() => {
                console.log(`Attempting to reconnect... (${this.reconnectAttempts}/${this.maxReconnectAttempts})`);
                this.connect();
            }, 3000);
        }
    }
    
    onConnectionOpen() {
        // Override in implementation
    }
    
    handleMessage(data) {
        // Override in implementation
        console.log('Received message:', data);
    }
    
    send(data) {
        if (this.socket && this.socket.readyState === WebSocket.OPEN) {
            this.socket.send(JSON.stringify(data));
        }
    }
    
    disconnect() {
        if (this.socket) {
            this.socket.close();
        }
    }
}

// Export for use in other scripts
window.VihangamWebSocket = VihangamWebSocket;
window.showNotification = showNotification;
window.setLoadingState = setLoadingState;
window.formatTime = formatTime;
window.formatDate = formatDate;