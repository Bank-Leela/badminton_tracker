const mongoose = require('mongoose');

const recordSchema = new mongoose.Schema({
    user: {
        type: mongoose.Schema.Types.ObjectId,
        required: true,
        ref: 'User',
    },
    date: {
        type: Date,
        required: true,
        default: Date.now,
    },
    opponent: {
        type: String,
        required: true,
        trim: true,
    },
    playerScore: {
        type: Number,
        required: true,
        min: 0,
    },
    opponentScore: {
        type: Number,
        required: true,
        min: 0,
    },
    result: {
        type: String,
        required: true,
        enum: ['Win', 'Loss', 'Draw'],
    },
    notes: {
        type: String,
        trim: true,
    }
}, { timestamps: true });

module.exports = mongoose.model('Record', recordSchema);
