const express = require('express');
const router = express.Router();
const Record = require('../models/Record');
const { protect } = require('../middleware/authMiddleware');

// @desc    Get all records for a user
// @route   GET /api/records
// @access  Private
router.get('/', protect, async (req, res) => {
  try {
    const records = await Record.find({ user: req.user.id }).sort({ date: -1 });
    res.status(200).json(records);
  } catch (error) {
    console.error('Error fetching records:', error);
    res.status(500).json({ message: 'Server Error' });
  }
});

// @desc    Add a new record
// @route   POST /api/records
// @access  Private
router.post('/', protect, async (req, res) => {
  try {
    const { date, opponent, playerScore, opponentScore, result, notes } = req.body;

    // Validate main required fields
    if (!opponent || playerScore === undefined || opponentScore === undefined || !result) {
      return res.status(400).json({ message: 'Please add all required fields' });
    }

    const record = await Record.create({
      user: req.user.id,
      date: date || Date.now(),
      opponent,
      playerScore,
      opponentScore,
      result,
      notes,
    });

    res.status(201).json(record);
  } catch (error) {
    console.error('Error adding record:', error);
    res.status(500).json({ message: 'Server Error' });
  }
});

// @desc    Delete a record
// @route   DELETE /api/records/:id
// @access  Private
router.delete('/:id', protect, async (req, res) => {
  try {
    const record = await Record.findById(req.params.id);

    if (!record) {
      return res.status(404).json({ message: 'Record not found' });
    }

    // Checking if the logged in user matches the record's user
    if (record.user.toString() !== req.user.id) {
      return res.status(401).json({ message: 'User not authorized' });
    }

    await record.deleteOne();

    res.status(200).json({ id: req.params.id });
  } catch (error) {
    console.error('Error deleting record:', error);
    res.status(500).json({ message: 'Server Error' });
  }
});

module.exports = router;
