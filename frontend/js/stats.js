/**
 * stats.js
 * Handle statistics tracking (FPS, counts, emotions)
 */

import state from './state.js';
import config from './config.js';

// DOM elements
let personCountElement, faceCountElement, fpsCounterElement, emotionsStatsElement, actionsStatsElement, detectedPeopleListElement;

/**
 * Initialize stats module
 * @param {Object} elements - DOM elements
 */
export function initStats(elements) {
    personCountElement = elements.personCountElement;
    faceCountElement = elements.faceCountElement;
    fpsCounterElement = elements.fpsCounterElement;
    emotionsStatsElement = document.getElementById('emotionsStats');
    actionsStatsElement = document.getElementById('actionsStats');
    detectedPeopleListElement = document.getElementById('detectedPeopleList').querySelector('.people-list-content');
}

/**
 * Setup FPS counter
 */
export function setupFpsCounter() {
    // Clear existing timer if any
    if (state.fpsTimerId) {
        clearInterval(state.fpsTimerId);
    }
    
    // Setup new timer
    state.frameCount = 0;
    state.lastFrameTime = performance.now();
    
    state.fpsTimerId = setInterval(() => {
        const currentTime = performance.now();
        const elapsedTime = (currentTime - state.lastFrameTime) / 1000;
        
        if (elapsedTime > 0) {
            const fps = Math.round(state.frameCount / elapsedTime);
            fpsCounterElement.textContent = fps;
            
            // Reset counters
            state.frameCount = 0;
            state.lastFrameTime = currentTime;
        }
    }, 1000);
    
    console.log("FPS counter started");
}

/**
 * Update statistics display
 * @param {number} personCount - Number of people detected
 * @param {number} faceCount - Number of faces detected
 * @param {Array} personBoxes - Array of detected person objects
 * @param {Array} faceBoxes - Array of detected face objects
 */
export function updateStats(personCount, faceCount, personBoxes, faceBoxes) {
    personCountElement.textContent = personCount;
    faceCountElement.textContent = faceCount;

    // Update emotion stats
    updateEmotionStats(faceBoxes);

    // Update action stats
    updateActionStats(personBoxes);

    // Update detected people list
    updateDetectedPeopleList(personBoxes, faceBoxes); // Add this line
}

/**
 * Update emotion statistics
 * @param {Array} faceBoxes - Face detection boxes with emotion data
 */
function updateEmotionStats(faceBoxes) {
    // Skip if element doesn't exist yet
    if (!emotionsStatsElement) return;
    
    // Count emotions
    const emotionCounts = {};
    
    faceBoxes.forEach(box => {
        if (box.emotion) {
            emotionCounts[box.emotion] = (emotionCounts[box.emotion] || 0) + 1;
        }
    });
    
    // Build HTML for emotion stats
    let emotionStatsHtml = '<h4>Cảm xúc:</h4>';
    
    if (Object.keys(emotionCounts).length === 0) {
        emotionStatsHtml += '<div class="emotion-stat-item">Không có dữ liệu</div>';
    } else {
        Object.entries(emotionCounts).forEach(([emotion, count]) => {
            const color = config.emotionColors[emotion] || config.emotionColor;
            emotionStatsHtml += `
                <div class="emotion-stat-item">
                    <span class="emotion-indicator" style="background-color: ${color}"></span>
                    <span class="emotion-name">${emotion}:</span>
                    <span class="emotion-count">${count}</span>
                </div>
            `;
        });
    }
    
    // Update the DOM
    emotionsStatsElement.innerHTML = emotionStatsHtml;
}

/**
 * Update action statistics
 * @param {Array} personBoxes - Person detection boxes with action data
 */
function updateActionStats(personBoxes) {
    // Skip if element doesn't exist yet
    if (!actionsStatsElement) return;
    
    // Count actions
    const actionCounts = {};
    
    personBoxes.forEach(box => {
        if (box.action) {
            actionCounts[box.action] = (actionCounts[box.action] || 0) + 1;
        }
    });
    
    // Build HTML for action stats
    let actionStatsHtml = '<h4>Hành vi:</h4>';
    
    if (Object.keys(actionCounts).length === 0) {
        actionStatsHtml += '<div class="action-stat-item">Không có dữ liệu</div>';
    } else {
        Object.entries(actionCounts).forEach(([action, count]) => {
            const color = config.actionColors[action] || config.actionColor;
            actionStatsHtml += `
                <div class="action-stat-item">
                    <span class="action-indicator" style="background-color: ${color}"></span>
                    <span class="action-name">${action}:</span>
                    <span class="action-count">${count}</span>
                </div>
            `;
        });
    }
    
    // Update the DOM
    actionsStatsElement.innerHTML = actionStatsHtml;
}

/**
 * Update the list of detected people and their details
 * @param {Array} personBoxes - Array of detected person objects
 * @param {Array} faceBoxes - Array of detected face objects
 */
export function updateDetectedPeopleList(personBoxes, faceBoxes) {
    if (!detectedPeopleListElement) return;

    let peopleHtml = '';

    if (personBoxes.length === 0 && faceBoxes.length === 0) {
        peopleHtml = '<div class="person-item">Không có người nào được phát hiện.</div>';
    } else {
        // Create a map to link faces to persons if possible
        const personFaceMap = new Map(); // Map: person_coords -> { person_box, face_boxes_list }

        // Group faces by their associated person (if any)
        faceBoxes.forEach(face => {
            // Find the person box that contains this face box
            const containingPerson = personBoxes.find(person => {
                const [px1, py1, px2, py2] = person.coords;
                const [fx1, fy1, fx2, fy2] = face.coords;
                return fx1 >= px1 && fy1 >= py1 && fx2 <= px2 && fy2 <= py2;
            });

            if (containingPerson) {
                const personKey = JSON.stringify(containingPerson.coords); // Use stringified coords as key
                if (!personFaceMap.has(personKey)) {
                    personFaceMap.set(personKey, { person: containingPerson, faces: [] });
                }
                personFaceMap.get(personKey).faces.push(face);
            } else {
                // If a face is detected but not inside any person box (e.g., small face, or person detection missed)
                // Treat it as a separate detected "person" for display purposes
                // Create a dummy person box for this face
                const dummyPerson = { coords: face.coords, confidence: 1.0, action: 'Không xác định' };
                const personKey = JSON.stringify(dummyPerson.coords);
                if (!personFaceMap.has(personKey)) {
                    personFaceMap.set(personKey, { person: dummyPerson, faces: [] });
                }
                personFaceMap.get(personKey).faces.push(face);
            }
        });

        // Handle persons without detected faces (YOLO person but no face)
        personBoxes.forEach(person => {
            const personKey = JSON.stringify(person.coords);
            if (!personFaceMap.has(personKey)) {
                // Add person if not already linked to a face
                personFaceMap.set(personKey, { person: person, faces: [] });
            }
        });


        // Build HTML for each detected person/face
        let personIndex = 1;
        personFaceMap.forEach(data => {
            const person = data.person;
            const faces = data.faces;

            let personDetails = [];

            // Add person action if available
            if (config.showActions && person.action) {
                personDetails.push(`Hành vi: ${person.action}`);
            }

            // Build a single line for each person/face
            let infoParts = [];
            
            // 1. Tên khuôn mặt (Nếu có)
            if (config.showFaceNames && faces.length > 0) {
                let faceNames = [];
                faces.forEach(face => {
                    if (face.similar_faces && face.similar_faces.length > 0) {
                        // join các chuỗi tên trực tiếp
                        faceNames.push(face.similar_faces.join(', ')); 
                    }
                });
                if (faceNames.length > 0) {
                    infoParts.push(`${faceNames.join(' & ')}`); // Vẫn join các nhóm tên từ các khuôn mặt khác nhau
                } else {
                    infoParts.push('Không xác định');
                }
            } else if (config.showFaces) {
                infoParts.push('Không xác định');
            }


            // 2. Hành vi (Nếu có)
            if (config.showActions && person.action) {
                infoParts.push(person.action);
            } else if (config.showActions && !person.action) {
                 infoParts.push('Hành vi: Không xác định');
            }
            
            // 3. Cảm xúc (Nếu có)
            if (config.showEmotions && faces.length > 0) {
                let emotions = [];
                faces.forEach(face => {
                    if (face.emotion) {
                        emotions.push(face.emotion);
                    }
                });
                if (emotions.length > 0) {
                    infoParts.push(emotions.join(', ')); // Join multiple emotions if multiple faces for one person
                } else {
                    infoParts.push('Cảm xúc: Không xác định');
                }
            } else if (config.showEmotions) {
                infoParts.push('Cảm xúc: Không xác định');
            }
            
            let personInfo = infoParts.join(' - ');
            if (personInfo === '') { // Fallback if no info at all
                personInfo = 'Không có thông tin chi tiết';
            }

            peopleHtml += `
                <div class="person-stat-item">
                    <span class="person-name">${personIndex++}. </span>
                    <span class="person-details">${personInfo}</span>
                </div>
            `;
        });
    }

    detectedPeopleListElement.innerHTML = peopleHtml;
}

/**
 * Reset statistics counters
 */
export function resetStats() {
    personCountElement.textContent = '0';
    faceCountElement.textContent = '0';
    fpsCounterElement.textContent = '0';
    
    // Clear emotion stats
    if (emotionsStatsElement) {
        emotionsStatsElement.innerHTML = '';
    }
    
    // Clear action stats
    if (actionsStatsElement) {
        actionsStatsElement.innerHTML = '';
    }

    // Clear detected people list
    if (detectedPeopleListElement) {
        detectedPeopleListElement.innerHTML = '';
    }
}